import asyncio
import requests
import json
import base64
import os
from playwright.async_api import async_playwright
from openai import OpenAI
import pdfplumber
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup

class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.start_time = None
        
    async def solve_quiz_chain(self, initial_url: str):
        """
        Solve a chain of quizzes starting from initial_url
        """
        import time
        self.start_time = time.time()
        current_url = initial_url
        max_iterations = 20  # Prevent infinite loops
        
        for i in range(max_iterations):
            # Check if we've exceeded 3 minutes
            if time.time() - self.start_time > 175:  # 175 seconds (leaving 5s buffer)
                print(f"Approaching time limit, stopping at iteration {i}")
                break
                
            print(f"\\nSolving quiz {i+1}: {current_url}")
            
            # Get the quiz question
            question_data = await self.fetch_quiz_page(current_url)
            
            if not question_data:
                print("Failed to fetch quiz page")
                break
            
            # Solve the quiz
            answer = await self.solve_question(question_data)
            
            # Submit the answer
            result = await self.submit_answer(
                question_data['submit_url'],
                current_url,
                answer
            )
            
            if not result:
                print("Failed to submit answer")
                break
            
            # Check if there's a next quiz
            if result.get('correct') and result.get('url'):
                current_url = result['url']
                print(f"Answer correct! Moving to next quiz: {current_url}")
            elif result.get('url') and not result.get('correct'):
                # Wrong answer but given next URL - can retry or skip
                print(f"Answer incorrect: {result.get('reason')}")
                # Option 1: Retry current question
                # Option 2: Skip to next (implemented below)
                current_url = result['url']
            elif result.get('correct'):
                print("Quiz chain completed successfully!")
                break
            else:
                print(f"Answer incorrect, no new URL: {result.get('reason')}")
                break
    
    async def fetch_quiz_page(self, url: str) -> dict:
        """
        Fetch and parse the quiz page using headless browser
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(url, wait_until='networkidle')
                await page.wait_for_timeout(2000)  # Wait for JS to execute
                
                # Get the rendered HTML
                content = await page.content()
                
                # Extract the question from the result div
                result_element = await page.query_selector("#result")
                if result_element:
                    question_html = await result_element.inner_html()
                else:
                    question_html = content
                
                await browser.close()
                
                # Parse the question
                soup = BeautifulSoup(question_html, 'html.parser')
                question_text = soup.get_text(strip=True)
                
                # Extract submit URL
                submit_url = self._extract_submit_url(question_text, content)
                
                # Extract any file URLs to download
                file_urls = self._extract_file_urls(soup)
                
                return {
                    'question': question_text,
                    'submit_url': submit_url,
                    'file_urls': file_urls,
                    'html': question_html
                }
                
            except Exception as e:
                print(f"Error fetching quiz page: {e}")
                await browser.close()
                return None
    
    def _extract_submit_url(self, text: str, html: str) -> str:
        """Extract the submit URL from question text"""
        # Look for URLs in the format https://example.com/submit
        urls = re.findall(r'https://[^\\s<>"]+/submit', text + html)
        return urls[0] if urls else None
    
    def _extract_file_urls(self, soup: BeautifulSoup) -> list:
        """Extract file download URLs from the question"""
        file_urls = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(ext in href for ext in ['.pdf', '.csv', '.xlsx', '.json', '.txt', '.png', '.jpg']):
                file_urls.append(href)
        return file_urls
    
    async def solve_question(self, question_data: dict) -> any:
        """
        Use LLM to understand and solve the question
        """
        question = question_data['question']
        file_urls = question_data.get('file_urls', [])
        
        # Download and process any files
        file_contents = {}
        for url in file_urls:
            content = await self._download_file(url)
            file_contents[url] = content
        
        # Build context for LLM
        context = f"Question: {question}\\n\\n"
        
        # Process files based on type
        processed_data = {}
        for url, content in file_contents.items():
            if '.pdf' in url:
                processed_data[url] = self._process_pdf(content)
            elif '.csv' in url:
                processed_data[url] = self._process_csv(content)
            elif '.json' in url:
                processed_data[url] = json.loads(content)
            elif url.endswith(('.png', '.jpg', '.jpeg')):
                processed_data[url] = self._process_image(content)
        
        # Add processed data to context
        if processed_data:
            context += "\\nData from files:\\n"
            for url, data in processed_data.items():
                context += f"\\n{url}:\\n{str(data)[:2000]}\\n"  # Limit length
        
        # Ask LLM to solve
        prompt = f"""You are a data analysis expert. Solve this question and provide ONLY the answer value.

{context}

Instructions:
- If the answer is a number, return just the number
- If the answer is a string, return just the string
- If the answer is a boolean, return true or false
- If the answer requires a chart/image, return "GENERATE_CHART" and I'll handle it
- If the answer requires a JSON object, return valid JSON

Answer:"""

        response = self.client.chat.completions.create(
            model="gpt-4o",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a precise data analyst. Return only the answer, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        answer_text = response.choices[0].message.content.strip()
        
        # Post-process answer
        answer = self._parse_answer(answer_text, question, processed_data)
        
        return answer
    
    async def _download_file(self, url: str):
        """Download file from URL"""
        response = requests.get(url)
        return response.content
    
    def _process_pdf(self, content: bytes) -> str:
        """Extract text and tables from PDF"""
        result = ""
        with pdfplumber.open(BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                result += f"\\n--- Page {i+1} ---\\n"
                result += page.extract_text() + "\\n"
                
                # Extract tables
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    result += f"\\nTable {j+1}:\\n"
                    df = pd.DataFrame(table[1:], columns=table[0])
                    result += df.to_string() + "\\n"
        
        return result
    
    def _process_csv(self, content: bytes) -> str:
        """Process CSV file"""
        df = pd.read_csv(BytesIO(content))
        return df.to_string()
    
    def _process_image(self, content: bytes) -> str:
        """Process image using vision model"""
        # Convert to base64
        img_base64 = base64.b64encode(content).decode('utf-8')
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail, extract any text, numbers, or data visible."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content
    
    def _parse_answer(self, answer_text: str, question: str, data: dict) -> any:
        """Parse the LLM's answer into the correct format"""
        # Try to parse as JSON
        try:
            return json.loads(answer_text)
        except:
            pass
        
        # Try to parse as number
        try:
            if '.' in answer_text:
                return float(answer_text)
            return int(answer_text)
        except:
            pass
        
        # Check if it's a boolean
        if answer_text.lower() in ['true', 'yes']:
            return True
        elif answer_text.lower() in ['false', 'no']:
            return False
        
        # Check if we need to generate a chart
        if "GENERATE_CHART" in answer_text or "chart" in question.lower():
            return self._generate_chart(data)
        
        # Return as string
        return answer_text
    
    def _generate_chart(self, data: dict) -> str:
        """Generate a chart and return as base64"""
        # Create a simple chart based on available data
        plt.figure(figsize=(10, 6))
        
        # Example: if we have CSV data, plot it
        for url, content in data.items():
            if isinstance(content, str) and 'DataFrame' in str(type(content)):
                df = pd.read_csv(BytesIO(content.encode()))
                df.plot(kind='bar')
                break
        
        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    async def submit_answer(self, submit_url: str, quiz_url: str, answer: any) -> dict:
        """Submit the answer to the quiz endpoint"""
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer
        }
        
        try:
            response = requests.post(submit_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Submit failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error submitting answer: {e}")
            return None
