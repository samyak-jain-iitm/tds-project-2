import asyncio
import requests
import json
import base64
import os
import time
import re
import logging
from openai import OpenAI
import pdfplumber
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class QuizSolver:
    def __init__(self, email: str, secret: str):
        """Initialize QuizSolver with credentials"""
        self.email = email
        self.secret = secret
        
        logger.info(f"🔧 Initializing QuizSolver for {email[:10]}...")
        
        openai_key = os.getenv("AI_PIPE_TOKEN")
        if not openai_key:
            logger.error("❌ AI_PIPE_TOKEN not set in environment")
            raise ValueError("AI_PIPE_TOKEN not set in environment")
        
        logger.info("✅ AI_PIPE_TOKEN found")
        
        try:
            self.client = OpenAI(api_key=openai_key, base_url="https://aipipe.org/openai/v1")
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.start_time = None
        
    async def solve_quiz_chain(self, initial_url: str):
        """
        Solve a chain of quizzes starting from initial_url
        """
        self.start_time = time.time()
        current_url = initial_url
        max_iterations = 20
        
        logger.info("="*80)
        logger.info(f"🚀 STARTING QUIZ CHAIN")
        logger.info(f"📍 Initial URL: {initial_url}")
        logger.info(f"⏱️  Max time: 170 seconds")
        logger.info("="*80)
        
        questions_solved = 0
        questions_correct = 0
        questions_wrong = 0
        
        for i in range(max_iterations):
            elapsed = time.time() - self.start_time
            if elapsed > 170:
                logger.warning(f"⏰ TIME LIMIT APPROACHING ({elapsed:.1f}s) - Stopping")
                break
            
            logger.info("")
            logger.info("="*80)
            logger.info(f"📝 QUIZ #{i+1}")
            logger.info(f"🔗 URL: {current_url}")
            logger.info(f"⏱️  Elapsed: {elapsed:.1f}s / 170s")
            logger.info("-"*80)
            
            try:
                # Fetch quiz page
                logger.info("🌐 Fetching quiz page...")
                question_data = await self.fetch_quiz_page(current_url)
                
                if not question_data:
                    logger.error("❌ Failed to fetch quiz page - Stopping")
                    break
                
                logger.info(f"✅ Quiz page fetched successfully")
                logger.info(f"❓ Question: {question_data['question'][:150]}...")
                logger.info(f"📎 Files found: {len(question_data.get('file_urls', []))}")
                
                # Solve the question
                logger.info("🤖 Solving question with LLM...")
                answer = await self.solve_question(question_data)
                logger.info(f"💡 Answer generated: {str(answer)[:100]}")
                
                # Submit answer
                logger.info("📤 Submitting answer...")
                result = await self.submit_answer(
                    question_data['submit_url'],
                    current_url,
                    answer
                )
                
                if not result:
                    logger.error("❌ Failed to submit answer - Stopping")
                    break
                
                questions_solved += 1
                
                # Check result
                if result.get('correct'):
                    questions_correct += 1
                    logger.info("✅ ✅ ✅ ANSWER CORRECT! ✅ ✅ ✅")
                    
                    if result.get('url'):
                        current_url = result['url']
                        logger.info(f"➡️  Moving to next quiz: {current_url}")
                    else:
                        logger.info("="*80)
                        logger.info("🎉 🎉 🎉 QUIZ CHAIN COMPLETED SUCCESSFULLY! 🎉 🎉 🎉")
                        logger.info("="*80)
                        break
                else:
                    questions_wrong += 1
                    reason = result.get('reason', 'No reason provided')
                    logger.warning(f"❌ ANSWER INCORRECT")
                    logger.warning(f"📋 Reason: {reason}")
                    
                    if result.get('url'):
                        current_url = result['url']
                        logger.info(f"➡️  Moving to next quiz anyway: {current_url}")
                    else:
                        logger.error("🛑 No next URL provided - Stopping")
                        break
                        
            except Exception as e:
                logger.error(f"💥 ERROR in quiz #{i+1}: {str(e)}")
                logger.exception("Full traceback:")
                break
        
        # Final summary
        total_time = time.time() - self.start_time
        logger.info("")
        logger.info("="*80)
        logger.info("🏁 QUIZ CHAIN ENDED")
        logger.info("-"*80)
        logger.info(f"📊 Questions solved: {questions_solved}")
        logger.info(f"✅ Correct answers: {questions_correct}")
        logger.info(f"❌ Wrong answers: {questions_wrong}")
        logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
        if questions_solved > 0:
            accuracy = (questions_correct / questions_solved) * 100
            logger.info(f"📈 Accuracy: {accuracy:.1f}%")
        logger.info("="*80)
    
    async def fetch_quiz_page(self, url: str) -> dict:
        """Fetch and parse quiz page using requests (no browser needed)"""
        try:
            logger.info(f"🌐 Fetching page with requests: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Page fetched: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            result_div = soup.find(id='result')
            if result_div:
                question_html = str(result_div)
                question_text = result_div.get_text(separator=' ', strip=True)
                logger.info("✅ Found #result element")
            else:
                question_html = response.text
                question_text = soup.get_text(separator=' ', strip=True)
                logger.info("⚠️  No #result element, using full page")
            
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'atob' in script.string:
                    logger.info("🔓 Found base64 encoded content, decoding...")
                    import base64
                    match = re.search(r'atob\(`([^`]+)`\)', script.string)
                    if match:
                        encoded = match.group(1)
                        decoded = base64.b64decode(encoded).decode('utf-8')
                        logger.info(f"✅ Decoded content: {decoded[:200]}...")
                        question_text = decoded
                        question_html = decoded
            
            submit_url = self._extract_submit_url(question_text, response.text)
            file_urls = self._extract_file_urls(soup, url)
            
            if submit_url:
                logger.info(f"✅ Submit URL found: {submit_url}")
            else:
                logger.warning("⚠️  No submit URL found")
            
            if file_urls:
                logger.info(f"📎 Found {len(file_urls)} file(s) to download:")
                for furl in file_urls:
                    logger.info(f"   • {furl}")
            
            return {
                'question': question_text,
                'submit_url': submit_url,
                'file_urls': file_urls,
                'html': question_html
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching page: {e}")
            logger.exception("Full traceback:")
            return None
    
    def _extract_submit_url(self, html: str, base_url: str = "") -> str:
        """
        Scrape the entire HTML for the submit URL.
        Looks at visible text, code/pre blocks, and hrefs.
        """
        soup = BeautifulSoup(html, "lxml")

        lines = []
        lines += soup.stripped_strings
        for tag in soup.find_all(['code', 'pre']):
            lines += tag.stripped_strings

        for tag in soup.find_all(['script']):
            if tag.string:
                lines.append(tag.string)

        alltext = "\n".join(lines)

        url_patterns = [
            r'post\s+\w+\s+to\s+(https?://[^\s"\'<>]+)',        
            r'post\s+\w+\s+to\s+(/[\w/\-]+)',                  
            r'(https?://[^\s"\'<>]+/submit)',                  
            r'(\/submit\b)',                                   
        ]

        for pat in url_patterns:
            m = re.search(pat, alltext, re.IGNORECASE)
            if m:
                url = m.group(1).strip()
                if url.startswith("/"):
                    if base_url:
                        submit_url = urljoin(base_url, url)
                        logger.info(f"✅ Relative submit URL found and joined: {submit_url}")
                        return submit_url
                    else:
                        logger.warning(f"⚠️ Found relative submit URL '{url}' but no base_url provided.")
                        return url
                else:
                    logger.info(f"✅ Absolute submit URL found: {url}")
                    return url

        for tag in soup.find_all('a', href=True):
            if 'submit' in tag['href']:
                href = tag['href']
                if href.startswith("/"):
                    submit_url = urljoin(base_url, href) if base_url else href
                else:
                    submit_url = href
                logger.info(f"✅ Found submit URL in anchor: {submit_url}")
                return submit_url

        logger.warning("⚠️ Could not extract submit URL from page content")
        return None

    
    def _extract_file_urls(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract downloadable file URLs"""
        file_urls = []
        extensions = ['.pdf', '.csv', '.xlsx', '.json', '.txt', '.png', '.jpg', '.jpeg']
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                url = href
            else:
                from urllib.parse import urljoin
                url = urljoin(base_url, href)
            
            if any(ext in url.lower() for ext in extensions):
                file_urls.append(url)
        
        logger.debug(f"📎 Extracted {len(file_urls)} file URL(s)")
        return file_urls
    
    async def solve_question(self, question_data: dict) -> any:
        """Use LLM to solve the question"""
        question = question_data['question']
        file_urls = question_data.get('file_urls', [])
        
        logger.info(f"🔍 Analyzing question ({len(question)} chars)")
        
        # Download and process files
        processed_data = {}
        if file_urls:
            logger.info(f"📥 Downloading and processing {len(file_urls)} file(s)...")
            
        for idx, url in enumerate(file_urls, 1):
            try:
                logger.info(f"   📥 [{idx}/{len(file_urls)}] Downloading: {url}")
                content = await self._download_file(url)
                logger.info(f"   ✅ Downloaded {len(content)} bytes")
                
                if '.pdf' in url.lower():
                    logger.info(f"   📄 Processing as PDF...")
                    processed_data[url] = self._process_pdf(content)
                elif '.csv' in url.lower():
                    logger.info(f"   📊 Processing as CSV...")
                    processed_data[url] = self._process_csv(content)
                elif '.json' in url.lower():
                    logger.info(f"   📋 Processing as JSON...")
                    processed_data[url] = json.loads(content)
                elif any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg']):
                    logger.info(f"   🖼️  Processing as Image...")
                    processed_data[url] = self._process_image(content)
                else:
                    logger.info(f"   📝 Processing as text...")
                    processed_data[url] = content.decode('utf-8', errors='ignore')
                
                logger.info(f"   ✅ File processed successfully")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing {url}: {e}")
        
        # Build context for LLM
        context = f"Question: {question}\n\n"
        
        if processed_data:
            context += "Data from files:\n"
            for url, data in processed_data.items():
                context += f"\nFile: {url}\n"
                data_str = str(data)
                context += data_str[:3000] + ("..." if len(data_str) > 3000 else "")
                context += "\n"
        
        prompt = f"""{context}

INSTRUCTIONS:
You are solving a quiz that will submit the following JSON:
{{
  "email": "{self.email}",
  "secret": "{self.secret}",
  "url": "{question_data.get('url','URL')}",
  "answer": (YOUR ANSWER GOES HERE)
}}

YOUR TASK:
Return ONLY the value for the "answer" key. The "answer" may need to be a boolean, number, string, base64 URI of a file attachment, or a JSON object with a combination of these.
- If the answer should be a number, output just the number (e.g., 42, 3.14)
- If the answer should be a string, output just the string, no quotes
- If the answer should be true or false, output true or false
- If the answer should be an object (explicitly requested), return a valid JSON object

Just output the value to use for "answer", and NOTHING ELSE.

Answer:
"""

        try:
            logger.info("🤖 Calling LLM (gpt-5-mini)...")
            logger.debug(f"   Context length: {len(prompt)} chars")
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a precise data analyst. Return only the answer, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )
            
            answer_text = response.choices[0].message.content.strip()
            logger.info(f"✅ LLM response received: {answer_text[:100]}")
            
            answer = self._parse_answer(answer_text, question, processed_data)
            logger.info(f"✅ Answer parsed as type: {type(answer).__name__}")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {e}")
            logger.exception("Full traceback:")
            return "Error"
    
    async def _download_file(self, url: str) -> bytes:
        """Download file from URL"""
        logger.debug(f"⬇️  Downloading: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        logger.debug(f"✅ Downloaded {len(response.content)} bytes")
        return response.content
    
    def _process_pdf(self, content: bytes) -> str:
        """Extract text and tables from PDF"""
        result = []
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                logger.debug(f"📄 PDF has {len(pdf.pages)} page(s)")
                
                for i, page in enumerate(pdf.pages, 1):
                    result.append(f"\n--- Page {i} ---")
                    
                    text = page.extract_text()
                    if text:
                        result.append(text)
                        logger.debug(f"   Page {i}: {len(text)} chars extracted")
                    
                    tables = page.extract_tables()
                    if tables:
                        logger.debug(f"   Page {i}: {len(tables)} table(s) found")
                    
                    for j, table in enumerate(tables, 1):
                        if table:
                            result.append(f"\nTable {j}:")
                            df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                            result.append(df.to_string())
                            logger.debug(f"   Table {j}: {df.shape[0]} rows, {df.shape[1]} cols")
                            
            logger.info(f"✅ PDF processed: {len(pdf.pages)} pages, {len(result)} sections")
                            
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {e}")
            result.append(f"Error processing PDF: {e}")
        
        return "\n".join(result)
    
    def _process_csv(self, content: bytes) -> str:
        """Process CSV file"""
        try:
            df = pd.read_csv(BytesIO(content))
            logger.info(f"✅ CSV processed: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.debug(f"   Columns: {list(df.columns)}")
            return f"Shape: {df.shape}\n\n{df.to_string()}"
        except Exception as e:
            logger.error(f"❌ Error processing CSV: {e}")
            return f"Error processing CSV: {e}"
    
    def _process_image(self, content: bytes) -> str:
        """Process image using vision model"""
        try:
            logger.info("🖼️  Encoding image to base64...")
            img_base64 = base64.b64encode(content).decode('utf-8')
            logger.debug(f"   Base64 length: {len(img_base64)} chars")
            
            logger.info("🤖 Calling vision model...")
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
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
                max_completion_tokens=500
            )
            
            result = response.choices[0].message.content
            logger.info(f"✅ Vision model response: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return f"Error processing image: {e}"
    
    def _parse_answer(self, answer_text: str, question: str, data: dict) -> any:
        """Parse LLM answer into correct format"""
        logger.debug(f"🔍 Parsing answer: {answer_text[:50]}...")
        
        # Try JSON
        try:
            parsed = json.loads(answer_text)
            logger.debug("   ✅ Parsed as JSON")
            return parsed
        except:
            pass
        
        # Try number
        try:
            if '.' in answer_text:
                parsed = float(answer_text)
                logger.debug(f"   ✅ Parsed as float: {parsed}")
                return parsed
            parsed = int(answer_text)
            logger.debug(f"   ✅ Parsed as int: {parsed}")
            return parsed
        except:
            pass
        
        # Try boolean
        if answer_text.lower() in ['true', 'yes']:
            logger.debug("   ✅ Parsed as boolean: True")
            return True
        elif answer_text.lower() in ['false', 'no']:
            logger.debug("   ✅ Parsed as boolean: False")
            return False
        
        # Return as string
        logger.debug("   ✅ Returned as string")
        return answer_text
    
    async def submit_answer(self, submit_url: str, quiz_url: str, answer: any) -> dict:
        """Submit answer to quiz endpoint"""
        if not submit_url:
            logger.error("❌ No submit URL found - cannot submit answer")
            return None
        
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer
        }
        
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.debug(f"   Payload: {json.dumps(payload, indent=2)[:200]}")
        
        try:
            response = requests.post(submit_url, json=payload, timeout=15)
            
            logger.info(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"   Response: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"❌ Submit failed: {response.status_code}")
                logger.error(f"   Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error submitting answer: {e}")
            logger.exception("Full traceback:")
            return None
