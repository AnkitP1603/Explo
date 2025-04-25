from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import stanza

app = FastAPI()

generation_model_id = "tiiuae/falcon-rw-1b"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_id, trust_remote_code=True)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_id, trust_remote_code=True)
text_generator = pipeline("text-generation", model=generation_model, tokenizer=generation_tokenizer)

sentiment_model_id = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_id)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_id)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

nlp = stanza.Pipeline(lang='en', dir='/models/stanza', processors='tokenize,pos,lemma,depparse')

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

class SentimentRequest(BaseModel):
    text: str

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": f"Models loaded: [Text Generation: '{generation_model_id}'], [Sentiment Analysis: '{sentiment_model_id}'], [Stanza: 'english']"
    }

@app.post("/generate")
def generate(request: GenerateRequest):
    output = text_generator(
        request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True
    )
    return {"response": output[0]["generated_text"]}

@app.post("/sentiment")
def sentiment(request: SentimentRequest):
    result = sentiment_analyzer(request.text)[0]
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    label = label_map.get(result['label'], "unknown")
    score = round(result['score'] * 100, 2)

    return {
        "sentiment": label,
        "confidence": f"{score}%"
    }

@app.post("/analyze")
def analyze_text(request: AnalyzeRequest):
    doc = nlp(request.text)
    results = []
    for sentence in doc.sentences:
        for word in sentence.words:
            results.append({
                "text": word.text,
                "lemma": word.lemma,
                "pos": word.upos,
                "head": word.head,
                "deprel": word.deprel
            })
    return {"tokens": results}


app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    with open("app/static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)
