# app.py

import os
import PyPDF2
import re
import logging
import dotenv
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

dotenv.load_dotenv()

# Config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PDF_PATH = os.environ.get("PDF_PATH", "static/docs/hr.pdf")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()))

app = Flask(__name__)
app.secret_key = SECRET_KEY

class GeminiClient:
    def __init__(self, api_key=GEMINI_API_KEY):
        self.api_key = api_key
        if not api_key:
            raise Exception("GEMINI_API_KEY not provided")
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self._check_connection()
        
    def _check_connection(self):
        """Check if Gemini API is accessible"""
        try:
            # Test the API with a simple request
            response = self.model.generate_content("Hello")
            if response.text:
                logging.info("Gemini API is ready and accessible")
            else:
                logging.warning("Gemini API responded but no text generated")
        except Exception as e:
            logging.error(f"Cannot connect to Gemini API: {e}")
            raise Exception(f"Gemini API not accessible: {e}")
        
    def generate(self, prompt, max_tokens=512, temperature=0.3):
        """Generate text using Gemini API"""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "No response generated from Gemini"
                
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"

class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.chunks = []
        self.gemini_client = GeminiClient()
        self.load_and_process_pdf()
        
        # Remove the static pattern definitions since we're using Gemini now
        # self.basic_patterns and self.hr_keywords are no longer needed

    def extract_text(self):
        text = ""
        with open(self.pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                try:
                    content = page.extract_text()
                    if content:
                        text += f"\n--- Page {i+1} ---\n{content}"
                except Exception as e:
                    logging.warning(f"Failed to extract text from page {i}: {e}")
        return text

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"--- Page \d+ ---", "", text)
        return text.strip()

    def chunk_text(self, text, chunk_size=800):
        sentences = re.split(r'[.!?]+', self.clean_text(text))
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence
        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if len(chunk) > 50]

    def load_and_process_pdf(self):
        logging.info("Processing PDF...")
        text = self.extract_text()
        if not text:
            raise ValueError("No text extracted from PDF")
        self.chunks = self.chunk_text(text)
        logging.info(f"Loaded {len(self.chunks)} chunks from PDF")

    def classify_user_intent(self, question):
        """Use Gemini to classify user intent and determine response strategy"""
        classification_prompt = f"""You are an intent classifier for an HR chatbot. Analyze the user's question and classify it into one of these categories:

1. GREETING - Simple greetings, hellos, good morning, etc.
2. CASUAL_CHAT - How are you, what's up, casual conversation not related to HR
3. GRATITUDE - Thank you, thanks, appreciation messages
4. GOODBYE - Bye, farewell, see you later, etc.
5. HR_QUERY - Any question related to HR policies, employee benefits, leave, company policies, work procedures, etc. (even with spelling mistakes)
6. NON_HR_QUERY - Questions not related to HR or company policies (weather, general knowledge, etc.)

User question: "{question}"

Respond with ONLY the category name (GREETING, CASUAL_CHAT, GRATITUDE, GOODBYE, HR_QUERY, or NON_HR_QUERY) and a brief reason in this format:
CATEGORY: [category]
REASON: [brief explanation]"""

        try:
            response = self.gemini_client.generate(classification_prompt, max_tokens=100, temperature=0.1)
            
            # Parse the response
            lines = response.strip().split('\n')
            category = None
            reason = None
            
            for line in lines:
                if line.startswith('CATEGORY:'):
                    category = line.replace('CATEGORY:', '').strip()
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
            
            return category, reason
            
        except Exception as e:
            logging.error(f"Intent classification error: {e}")
            # Fallback to HR_QUERY if classification fails
            return "HR_QUERY", "Classification failed, defaulting to HR query"

    def generate_contextual_response(self, question, category):
        """Generate appropriate response based on classified intent"""
        response_prompts = {
            'GREETING': f"""The user said: "{question}"
            
            Respond as a friendly HR assistant with a warm greeting. Keep it professional but welcoming. Mention that you're here to help with HR questions.""",
            
            'CASUAL_CHAT': f"""The user said: "{question}"
            
            Respond in a friendly, professional manner as an HR assistant. Keep the conversation light but redirect gently to HR topics you can help with.""",
            
            'GRATITUDE': f"""The user said: "{question}"
            
            Respond appropriately to their thanks as an HR assistant. Be gracious and offer continued assistance with HR matters.""",
            
            'GOODBYE': f"""The user said: "{question}"
            
            Respond with a professional but warm farewell as an HR assistant. Invite them to return with any HR questions.""",
            
            'NON_HR_QUERY': f"""The user asked: "{question}"
            
            Politely explain that you're specifically designed to help with HR-related questions and company policies. Encourage them to ask about HR topics like leave policies, benefits, procedures, etc. Be helpful but redirect to your purpose."""
        }
        
        if category in response_prompts:
            try:
                response = self.gemini_client.generate(response_prompts[category], max_tokens=150, temperature=0.7)
                return response.strip()
            except Exception as e:
                logging.error(f"Response generation error: {e}")
                return "I'm here to help with your HR questions. How can I assist you today?"
        
        return "How can I help you with HR-related questions today?"

    def find_relevant_chunks(self, query, top_k=3):
        """Simple keyword-based relevance scoring"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            # Simple overlap scoring
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(query_words)
                scored_chunks.append((chunk, score))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def answer_question(self, question):
        # First, use Gemini to classify the user's intent
        category, reason = self.classify_user_intent(question)
        logging.info(f"Question classified as: {category}, Reason: {reason}")
        
        # Handle non-HR queries with contextual responses
        if category in ['GREETING', 'CASUAL_CHAT', 'GRATITUDE', 'GOODBYE', 'NON_HR_QUERY']:
            return {
                "answer": self.generate_contextual_response(question, category),
                "sources": [],
                "confidence": 1.0,
                "type": category.lower(),
                "classification_reason": reason
            }
        
        # For HR queries, search the PDF
        if category == 'HR_QUERY':
            chunks = self.find_relevant_chunks(question)
            if not chunks:
                return {
                    "answer": "I couldn't find relevant information about this topic in the HR documents. Please contact your HR department directly for assistance, or try rephrasing your question with different keywords.",
                    "sources": [],
                    "confidence": 0,
                    "type": "no_relevant_content",
                    "classification_reason": reason
                }
            
            context = "\n\n".join([c for c, _ in chunks])
            
            prompt = f"""You are an HR assistant that answers questions based ONLY on the provided HR policy context. Follow these rules strictly:

1. Use ONLY the information from the context provided below
2. If the context doesn't contain enough information to answer the question, clearly state that the information is not available in the HR documents
3. Do not make up or assume any information not present in the context
4. Be specific and reference the relevant sections when possible
5. If asked about company-specific policies, refer only to what's in the context
6. Handle spelling mistakes or variations in the user's question gracefully

Context from HR Documents:
{context}

User Question: {question}

Please provide a clear and accurate answer based ONLY on the HR policy context above. If the information is not available in the context, clearly state that."""
            
            try:
                response = self.gemini_client.generate(prompt, max_tokens=512, temperature=0.2)
                return {
                    "answer": response.strip(),
                    "sources": [c[:200] + "..." for c, _ in chunks],
                    "confidence": sum([s for _, s in chunks]) / len(chunks) if chunks else 0,
                    "type": "hr_query",
                    "classification_reason": reason
                }
            except Exception as e:
                logging.error(f"Gemini error: {e}")
                return {
                    "answer": "I'm having trouble processing your question right now. Please try again or contact your HR department directly.",
                    "sources": [],
                    "confidence": 0,
                    "type": "error",
                    "classification_reason": reason
                }
        
        # Fallback for any unhandled cases
        return {
            "answer": "I'm not sure how to handle that request. Could you please rephrase your question or ask me about HR policies and procedures?",
            "sources": [],
            "confidence": 0.5,
            "type": "unknown",
            "classification_reason": reason
        }

# Initialize chatbot
try:
    chatbot = PDFChatbot(PDF_PATH)
    logging.info("Chatbot initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize chatbot: {e}")
    chatbot = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500
        
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    
    try:
        result = chatbot.answer_question(question)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return jsonify({"error": "Failed to process question"}), 500

@app.route("/api/status")
def status():
    if not chatbot:
        return jsonify({"status": "error", "message": "Chatbot not initialized"}), 500
    
    try:
        test_response = chatbot.gemini_client.generate("Hello", max_tokens=5)
        gemini_status = "connected" if test_response else "error"
    except Exception as e:
        gemini_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "ready",
        "chunks": len(chatbot.chunks),
        "gemini_status": gemini_status,
        "model": "gemini-1.5-flash"
    })

if __name__ == "__main__":
    app.run(debug=True)
