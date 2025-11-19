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
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("⚠️ Selenium not installed - JS-rendered pages may not work")


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
        self.driver = None

    def __del__(self):
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except:
                pass

    def _init_selenium(self):
        if not SELENIUM_AVAILABLE:
            return False
        
        if hasattr(self, 'driver') and self.driver:
            return True
        
        try:
            options = Options()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/google-chrome-stable")
            options.binary_location = chrome_bin
            logger.info(f"🔧 Using Chrome binary at: {chrome_bin}")

            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service

            service = Service(ChromeDriverManager().install())
            
            self.driver = webdriver.Chrome(service=service, options=options)
            logger.info("✅ Selenium WebDriver initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Selenium: {e}")
            logger.exception("Full traceback:")
            return False

    def _scrape_with_selenium(self, url: str) -> str:
        if not self._init_selenium():
            return None
        
        try:
            logger.info(f"   🌐 Using Selenium to render: {url}")
            self.driver.get(url)
            
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(2)
            
            try:
                question_elem = self.driver.find_element(By.ID, "question")
                content = question_elem.text
            except:
                content = self.driver.find_element(By.TAG_NAME, "body").text
            
            logger.info(f"   ✅ Selenium rendered {len(content)} chars")
            return content
            
        except Exception as e:
            logger.error(f"   ❌ Selenium scraping failed: {e}")
            return None

        
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
        try:
            logger.info(f"🌐 Fetching page with requests: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Page fetched: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            full_html = response.text
            
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
                    logger.info("🔓 Found script with atob, extracting base64...")
                    
                    base64_candidates = re.findall(r'`([A-Za-z0-9+/=]{50,})`', script.string)
                    
                    for candidate in base64_candidates:
                        try:
                            decoded = base64.b64decode(candidate).decode('utf-8')
                            logger.info(f"✅ Decoded {len(decoded)} chars")
                            question_text = decoded
                            question_html = decoded
                            break
                        except Exception:
                            continue
            
            if '<span id="cutoff"></span>' in full_html or 'emailNumber()' in full_html:
                logger.info("🔍 Detected dynamic content, fetching with Selenium...")
                if self._init_selenium():
                    try:
                        self.driver.get(url)
                        time.sleep(3)  # Wait for JS to execute
                        
                        # Get the rendered page text
                        rendered_body = self.driver.find_element(By.TAG_NAME, "body").text
                        question_text += f"\n\nRendered page content:\n{rendered_body}"
                        logger.info(f"✅ Added rendered content: {len(rendered_body)} chars")
                    except Exception as e:
                        logger.error(f"⚠️ Failed to render with Selenium: {e}")
            
            submit_url = self._extract_submit_url(question_html, url)
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
                'html': question_html,
                'url': url
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching page: {e}")
            logger.exception("Full traceback:")
            return None

    
    def _extract_submit_url(self, html: str, base_url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        
        logger.debug(f"Searching for submit URL in {len(html)} chars of HTML")
        logger.debug(f"Base URL: {base_url}")
        
        links_found = soup.find_all('a', href=True)
        logger.debug(f"Found {len(links_found)} total links")
        
        for link in links_found:
            href = link['href']
            logger.debug(f"Checking link: {href}")
            if 'submit' in href.lower():
                if href.startswith('http'):
                    submit_url = href
                else:
                    submit_url = urljoin(base_url, href)
                logger.info(f"✅ Found submit URL in hyperlink: {submit_url}")
                return submit_url
        
        all_text = soup.get_text(separator=' ')
        
        absolute_patterns = [
            r'https?://[^\s<>"\']+/submit\b',
            r'POST\s+.*?\s+to\s+(https?://[^\s<>"\']+)',
        ]
        
        for pattern in absolute_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                url = matches[0].strip().rstrip('/')
                if not url.endswith('/submit'):
                    url += '/submit'
                logger.info(f"✅ Found absolute submit URL in text: {url}")
                return url
        
        for elem in soup.find_all(string=re.compile(r'POST\s+.*?\s+back\s+to', re.I)):
            parent = elem.parent
            all_siblings = [elem]
            sib = elem.next_sibling
            while sib:
                if isinstance(sib, NavigableString):
                    all_siblings.append(str(sib))
                elif isinstance(sib, Tag):
                    all_siblings.append(sib.get_text())
                sib = sib.next_sibling
            
            combined = ''.join(all_siblings).strip()
            
            m = re.search(r'(https?://[^\s<>"\']+)', combined)
            if m:
                domain = m.group(1).rstrip('/')
                if '/submit' in combined:
                    url = domain + '/submit' if not domain.endswith('/submit') else domain
                    logger.info(f"✅ Assembled submit URL from siblings: {url}")
                    return url
        
        if '/submit' in all_text and base_url and base_url.startswith('http'):
            url = urljoin(base_url, '/submit')
            logger.info(f"✅ Fallback: joined /submit with base URL: {url}")
            return url
        
        logger.warning("⚠️ Could not extract submit URL")
        return None

    
    def _extract_file_urls(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract downloadable file URLs"""
        file_urls = []
        extensions = ['.pdf', '.csv', '.xlsx', '.json', '.txt', '.png', '.jpg', '.jpeg', '.opus', '.mp3', '.wav', '.m4a']
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                url = href
            else:
                from urllib.parse import urljoin
                url = urljoin(base_url, href)
            
            if any(ext in url.lower() for ext in extensions):
                file_urls.append(url)
        
        for audio in soup.find_all('audio', src=True):
            src = audio['src']
            if src.startswith('http'):
                url = src
            else:
                url = urljoin(base_url, src)
            file_urls.append(url)
        
        logger.debug(f"📎 Extracted {len(file_urls)} file URL(s)")
        return file_urls
    
    def _process_audio(self, content: bytes) -> str:
        import speech_recognition as sr
        import os
        import tempfile
        from pydub import AudioSegment 

        try:
            logger.info("🎵 Using Google Web Speech (Cloud)...")
            
            # 1. Write original content (likely opus/mp3) to file
            with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as temp_opus:
                temp_opus.write(content)
                opus_path = temp_opus.name

            # 2. Convert to WAV (Required by SpeechRecognition) using ffmpeg (installed in Docker)
            wav_path = opus_path + ".wav"
            os.system(f"ffmpeg -i {opus_path} -ar 16000 -ac 1 {wav_path} -y -hide_banner -loglevel error")

            # 3. Send to Google
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = r.record(source)
                # This uses Google's free text-to-speech API (no key needed)
                text = r.recognize_google(audio_data)
                
            logger.info(f"✅ Google Transcription: {text}")
            
            # Cleanup
            os.unlink(opus_path)
            os.unlink(wav_path)
            return text

        except Exception as e:
            logger.error(f"❌ Google Speech failed: {e}")
            return "[Audio transcription failed]"

    async def solve_question(self, question_data: dict) -> any:
        question = question_data['question']
        file_urls = question_data.get('file_urls', [])
        current_url = question_data.get('url', '')
        
        logger.info(f"🔍 Analyzing question ({len(question)} chars)")
        
        # AUTO-DETECT URLs to scrape from question
        soup_question = BeautifulSoup(question, 'html.parser')
        scrape_links = []
        for link in soup_question.find_all('a', href=True):
            href = link['href']
            
            # Skip submit URLs and file downloads
            if 'submit' in href.lower():
                logger.debug(f"   Skipping submit URL: {href}")
                continue
            
            if any(ext in href.lower() for ext in ['.pdf', '.csv', '.xlsx', '.json', '.txt', '.png', '.jpg', '.jpeg']):
                logger.debug(f"   Skipping file URL: {href}")
                continue
            
            # Build full URL
            if href.startswith('http'):
                full_url = href
            else:
                full_url = urljoin(current_url, href)
            
            # Replace $EMAIL placeholder with actual email
            full_url = full_url.replace('$EMAIL', self.email)
            
            scrape_links.append(full_url)
            logger.debug(f"   Will scrape: {full_url}")

        if scrape_links:
            logger.info(f"🔗 Found {len(scrape_links)} URL(s) to scrape from question")
            for scrape_url in scrape_links:
                try:
                    logger.info(f"   🌐 Scraping: {scrape_url}")
                    scrape_response = requests.get(scrape_url, timeout=30)
                    
                    scrape_soup = BeautifulSoup(scrape_response.text, 'html.parser')
                    scrape_text = scrape_soup.get_text(separator=' ', strip=True)
                    
                    if len(scrape_text) < 20:
                        logger.warning(f"   ⚠️  Content too short, trying Selenium...")
                        selenium_content = self._scrape_with_selenium(scrape_url)
                        if selenium_content and len(selenium_content) > 20:
                            scrape_text = selenium_content

                    file_urls.append(('scraped_page', scrape_text))
                    logger.info(f"   ✅ Scraped {len(scrape_text)} chars")
                except Exception as e:
                    logger.error(f"   ❌ Failed to scrape {scrape_url}: {e}")

        
        # Download and process files
        processed_data = {}
        if file_urls:
            logger.info(f"📥 Downloading and processing {len(file_urls)} file(s)...")
            
        for idx, item in enumerate(file_urls, 1):
            try:
                if isinstance(item, tuple):
                    url, content_text = item
                    processed_data[url] = content_text
                    logger.info(f"   ✅ Added scraped content: {len(content_text)} chars")
                    continue
                    
                url = item
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
                elif any(ext in url.lower() for ext in ['.opus', '.mp3', '.wav', '.m4a']):  # ADD THIS
                    logger.info(f"   🎵 Processing as Audio...")
                    processed_data[url] = self._process_audio(content)
                else:
                    logger.info(f"   📝 Processing as text...")
                    processed_data[url] = content.decode('utf-8', errors='ignore')
                
                logger.info(f"   ✅ File processed successfully")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing {url}: {e}")
        
        # Build context
        context = f"Question: {question}\n\n"
        
        if processed_data:
            context += "Data provided:\n"
            for url, data in processed_data.items():
                context += f"\n=== Source: {url} ===\n"
                context += str(data)
                context += "\n"
        
        prompt = f"""{context}

        CRITICAL INSTRUCTIONS:
        1. Analyze the question and ALL data provided above
        2. If you see links to scrape, the content has already been fetched for you
        3. If data contains numbers to calculate (sum, mean, count, etc.), perform the calculation
        4. Return ONLY the final answer VALUE - NOT a JSON object

        EXAMPLES:
        - Question asks for a sum → Return: 12345 (just the number)
        - Question asks for text → Return: the resultant text (just the text, no quotes)
        - Question asks for true/false → Return: true (or false)
        - Question explicitly asks for JSON array → Return: [{{"x":1}}] (only if explicitly requested)

        DO NOT return:
        - {{"answer": 12345}} ❌
        - "the answer is 12345" ❌
        - Any explanations ❌

        The quiz server expects this format:
        {{
        "email": "{self.email}",
        "secret": "{self.secret}",
        "url": "quiz_url",
        "answer": YOUR_VALUE_HERE
        }}

        I will put your response directly into the "answer" field.

        Answer (just the value):
        """

        logger.info(f"📏 Prompt length: {len(prompt)} chars")
        logger.info(f"Full prompt:\n{prompt}")

        try:
            logger.info("🤖 Calling LLM (gpt-5-mini)...")
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a precise data analyst. Analyze all data thoroughly and return only the answer."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )
            
            answer_text = response.choices[0].message.content.strip()

            if not answer_text or answer_text == "":
                logger.error("❌ LLM returned EMPTY response!")
                logger.error(f"   Finish reason: {response.choices[0].finish_reason}")
                
                if response.choices[0].finish_reason == 'length':
                    logger.warning("⚠️ Hit token limit, retrying with increased limit...")
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=1000  
                    )
                    answer_text = response.choices[0].message.content.strip()
                
                if not answer_text:
                    return None
            
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
            # Try reading with header first
            df = pd.read_csv(BytesIO(content))
            
            # Check if first row looks like data (all numeric) rather than headers
            first_row_numeric = all(str(col).replace('.','').replace('-','').isdigit() for col in df.columns)
            
            if first_row_numeric:
                # Re-read without header
                df = pd.read_csv(BytesIO(content), header=None)
                logger.info(f"✅ CSV processed (no header): {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                logger.info(f"✅ CSV processed: {df.shape[0]} rows, {df.shape[1]} columns")
            
            logger.debug(f"   Columns: {list(df.columns)}")
            
            # Build output with each column as a list
            output = f"CSV Format (Shape: {df.shape}):\n"
            
            for col in df.columns:
                values = df[col].tolist()
                output += f"Column_{col}: {values}\n"
            
            return output
            
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
