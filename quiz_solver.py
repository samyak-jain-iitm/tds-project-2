import asyncio
import requests
import json
import base64
import os
import time
import re
from playwright.async_api import async_playwright
from openai import OpenAI
import pdfplumber
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Important: Use non-GUI backend for server
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        
        # Initialize OpenAI client
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        self.client = OpenAI(api_key=openai_key)
        self.start_time = None
        
    async def solve_quiz_chain(self, initial_url: str):
        """
        Solve a chain of quizzes starting from initial_url
        """
        self.start_time = time.time()
        current_url = initial_url
        max_iterations = 20
        
        print(f"Starting quiz chain from: {initial_url}")
        
        for i in range(max_iterations):
            # Check time limit (3 minutes = 180 seconds, use 170 for safety)
            elapsed = time.time() - self.start_time
            if elapsed > 170:
                print(f"Time limit approaching ({elapsed:.1f}s), stopping")
                break
                
            print(f"\n{'='*60}")
            print(f"Quiz {i+1}: {current_url}")
            print(f"Elapsed time: {elapsed:.1f}s")
            
            try:
                # Fetch quiz page
                question_data = await self.fetch_quiz_page(current_url)
                if not question_data:
                    print("Failed to fetch quiz page")
                    break
                
                print(f"Question: {question_data['question'][:100]}...")
                
                # Solve the question
                answer = await self.solve_question(question_data)
                print(f"Answer generated: {str(answer)[:100]}")
                
                # Submit answer
                result = await self.submit_answer(
                    question_data['submit_url'],
                    current_url,
                    answer
                )
                
                if not result:
                    print("Failed to submit answer")
                    break
                
                # Check result
                if result.get('correct'):
                    print("✅ Answer correct!")
                    if result.get('url'):
                        current_url = result['url']
                        print(f"Next quiz: {current_url}")
                    else:
                        print("🎉 Quiz chain completed!")
                        break
                else:
                    print(f"❌ Answer incorrect: {result.get('reason', 'No reason given')}")
                    if result.get('url'):
                        # Can retry or skip to next
                        current_url = result['url']
                        print(f"Moving to next: {current_url}")
                    else:
                        print("No next URL, stopping")
                        break
                        
            except Exception as e:
                print(f"Error in quiz {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\nQuiz solving ended. Total time: {time.time() - self.start_time:.1f}s")
    
    async def fetch_quiz_page(self, url: str) -> dict:
        """Fetch and parse quiz page using headless browser"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto(url, wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(2000)
                
                # Get rendered content
                content = await page.content()
                
                # Try to get result div
                result_element = await page.query_selector("#result")
                if result_element:
                    question_html = await result_element.inner_html()
                else:
                    question_html = content
                
                await browser.close()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(question_html, 'html.parser')
                question_text = soup.get_text(separator=' ', strip=True)
                
                # Extract submit URL
                submit_url = self._extract_submit_url(question_text, content)
                
                # Extract file URLs
                file_urls = self._extract_file_urls(soup, url)
                
                return {
                    'question': question_text,
                    'submit_url': submit_url,
                    'file_urls': file_urls,
                    'html': question_html
                }
                
        except Exception as e:
            print(f"Error fetching page: {e}")
            return None
    
    def _extract_submit_url(self, text: str, html: str) -> str:
        """Extract submit URL from question"""
        # Look for submit endpoint
        patterns = [
            r'https://[^\s<>"\']+/submit',
            r'Post.*?to\s+(https://[^\s<>"\']+)',
            r'submit.*?(https://[^\s<>"\']+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text + html, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def _extract_file_urls(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract downloadable file URLs"""
        file_urls = []
        extensions = ['.pdf', '.csv', '.xlsx', '.json', '.txt', '.png', '.jpg', '.jpeg']
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle relative URLs
            if href.startswith('http'):
                url = href
            else:
                from urllib.parse import urljoin
                url = urljoin(base_url, href)
            
            if any(ext in url.lower() for ext in extensions):
                file_urls.append(url)
        
        return file_urls
    
    async def solve_question(self, question_data: dict) -> any:
        """Use LLM to solve the question"""
        question = question_data['question']
        file_urls = question_data.get('file_urls', [])
        
        # Download and process files
        processed_data = {}
        for url in file_urls:
            try:
                content = await self._download_file(url)
                
                if '.pdf' in url.lower():
                    processed_data[url] = self._process_pdf(content)
                elif '.csv' in url.lower():
                    processed_data[url] = self._process_csv(content)
                elif '.json' in url.lower():
                    processed_data[url] = json.loads(content)
                elif any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg']):
                    processed_data[url] = self._process_image(content)
                else:
                    processed_data[url] = content.decode('utf-8', errors='ignore')
            except Exception as e:
                print(f"Error processing {url}: {e}")
        
        # Build prompt for LLM
        context = f"Question: {question}\n\n"
        
        if processed_data:
            context += "Data from files:\n"
            for url, data in processed_data.items():
                context += f"\nFile: {url}\n"
                data_str = str(data)
                context += data_str[:3000] + ("..." if len(data_str) > 3000 else "")
                context += "\n"
        
        prompt = f"""{context}

Analyze the question and data carefully. Provide ONLY the answer in the appropriate format:
- For numbers: return just the number (e.g., 42 or 3.14)
- For strings: return just the text
- For booleans: return true or false
- For JSON: return valid JSON object
- For images/charts: return "CHART_NEEDED"

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise data analyst. Return only the answer, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            answer_text = response.choices[0].message.content.strip()
            answer = self._parse_answer(answer_text, question, processed_data)
            
            return answer
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Error"
    
    async def _download_file(self, url: str) -> bytes:
        """Download file from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    
    def _process_pdf(self, content: bytes) -> str:
        """Extract text and tables from PDF"""
        result = []
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    result.append(f"\n--- Page {i} ---")
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        result.append(text)
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables, 1):
                        if table:
                            result.append(f"\nTable {j}:")
                            df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                            result.append(df.to_string())
        except Exception as e:
            result.append(f"Error processing PDF: {e}")
        
        return "\n".join(result)
    
    def _process_csv(self, content: bytes) -> str:
        """Process CSV file"""
        try:
            df = pd.read_csv(BytesIO(content))
            return f"Shape: {df.shape}\n\n{df.to_string()}"
        except Exception as e:
            return f"Error processing CSV: {e}"
    
    def _process_image(self, content: bytes) -> str:
        """Process image using vision model"""
        try:
            img_base64 = base64.b64encode(content).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image and extract all visible text, numbers, and data."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing image: {e}"
    
    def _parse_answer(self, answer_text: str, question: str, data: dict) -> any:
        """Parse LLM answer into correct format"""
        # Try JSON
        try:
            return json.loads(answer_text)
        except:
            pass
        
        # Try number
        try:
            if '.' in answer_text:
                return float(answer_text)
            return int(answer_text)
        except:
            pass
        
        # Try boolean
        if answer_text.lower() in ['true', 'yes']:
            return True
        elif answer_text.lower() in ['false', 'no']:
            return False
        
        # Return as string
        return answer_text
    
    async def submit_answer(self, submit_url: str, quiz_url: str, answer: any) -> dict:
        """Submit answer to quiz endpoint"""
        if not submit_url:
            print("No submit URL found")
            return None
        
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer
        }
        
        try:
            response = requests.post(submit_url, json=payload, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Submit failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error submitting: {e}")
            return None
