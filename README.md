# 🚀 LLM Microservice with FastAPI

A lightweight, Dockerized microservice that combines:

- 🔮 **Text Generation** using [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b)
- 💬 **Sentiment Analysis** using [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- 🌐 Simple HTML frontend to interact with both APIs

---

## 🛠️ Features

- FastAPI backend with REST endpoints
- Text generation and sentiment analysis via Hugging Face Transformers
- Clean frontend at `/ui`
- Containerized with Docker for easy deployment
- Accepts prompts and returns generated text or sentiment + confidence score
- Preloaded English NLP model with Stanza
---

## 📂 Project Structure

```
llm-microservice/  
  ├── app/|
  |       ├── main.py # FastAPI backend   
  |       └── static/|   
  |                  └── index.html # Frontend interface   
  ├── Dockerfile # Container config   
  ├── requirements.txt # Python dependencies  
  ├── .gitignore   
  └── README.md  
```

## 🚀 Run the App

### 🐋 Docker (recommended)

```bash
docker build -t llm-service .
docker run -p 8000:8000 llm-service
```
Then open in browser: http://localhost:8000/ui  

🔗 API Endpoints:  
➤ /generate  
➤ /sentiment    
➤ /analyze

✨ Technologies Used  
- Python 3.11
- FastAPI
- Hugging Face Transformers
- Docker
- HTML/CSS (for frontend)

📦 Install Locally (without Docker):  
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit: `http://127.0.0.1:8000/ui`


