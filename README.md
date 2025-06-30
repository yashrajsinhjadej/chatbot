# 🤖 AI PDF Chatbot with Gemini

A simple AI-powered chatbot that answers questions from PDF documents using Google's Gemini AI.

## 🚀 Features

- **AI-Powered Q&A**: Uses Google Gemini AI for intelligent responses
- **PDF Processing**: Analyze any PDF document
- **Semantic Search**: Find relevant content using advanced embeddings
- **Easy Deployment**: One-click deployment to Vercel

## 🛠️ Setup Instructions

### 1. Get Your Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

### 2. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### 3. Deploy to Vercel
1. Push code to GitHub
2. Connect GitHub repo to Vercel
3. Add environment variables in Vercel dashboard
4. Deploy!

## � Configuration

Required environment variables:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_secret_key_here
```