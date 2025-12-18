# sentiment.py
import logging
logging.basicConfig(level=logging.INFO)

def get_sentiment_pipeline():
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        model_name = "yiyanghkust/finbert-tone"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
        return nlp
    except Exception:
        # fallback to VADER
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        def vader_pipe(texts):
            out = []
            for t in texts:
                s = analyzer.polarity_scores(t)
                label = "neutral"
                if s['compound'] >= 0.05:
                    label = "positive"
                elif s['compound'] <= -0.05:
                    label = "negative"
                out.append({"label":label,"score":float(s['compound'])})
            return out
        return vader_pipe

_pipeline = None

def analyze_texts(list_of_texts):
    global _pipeline
    if _pipeline is None:
        _pipeline = get_sentiment_pipeline()
    return _pipeline(list_of_texts)
