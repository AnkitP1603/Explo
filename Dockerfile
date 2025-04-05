FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    -f https://download.pytorch.org/whl/torch_stable.html


RUN python3 -c "\
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification; \
print('Downloading Falcon model...'); \
pipeline('text-generation', model='tiiuae/falcon-rw-1b', trust_remote_code=True); \
print('Downloading CardiffNLP model...'); \
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')"


COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]