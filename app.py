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
        chunks = self.find_relevant_chunks(question)
        if not chunks:
            return {"answer": "No relevant content found.", "sources": [], "confidence": 0}
        
        context = "\n\n".join([c for c, _ in chunks])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above."""
        
        try:
            response = self.gemini_client.generate(prompt, max_tokens=512, temperature=0.3)
            return {
                "answer": response.strip(),
                "sources": [c[:200] + "..." for c, _ in chunks],
                "confidence": sum([s for _, s in chunks]) / len(chunks) if chunks else 0
            }
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            return {"answer": "Error generating response from Gemini model.", "sources": [], "confidence": 0}

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

# For Vercel deployment
app_instance = app