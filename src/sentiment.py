from typing import List

# Try FinBERT from transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    _tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    _model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    _use_finbert = True
except Exception:
    _use_finbert = False

# VADER fallback
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

def finbert_sentiment(texts: List[str]):
    if not _use_finbert:
        # If FinBERT not loaded, fallback to VADER for all texts
        return [vader_sentiment(t) for t in texts]
    inputs = _tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    outputs = _model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # FinBERT labels: 0-neutral,1-positive,2-negative (check model docs)
    # You can compute sentiment score as pos - neg prob
    scores = probs.detach().cpu().numpy()
    sentiment_scores = []
    for score in scores:
        sentiment_scores.append(float(score[1]) - float(score[2]))
    return sentiment_scores

def vader_sentiment(text: str) -> float:
    s = _vader.polarity_scores(text)
    return s['compound']

if __name__ == '__main__':
    test_text = "The company's earnings report was disappointing, but future outlook is positive."
    if _use_finbert:
        print("FinBERT sentiment:", finbert_sentiment([test_text]))
    else:
        print("VADER sentiment:", vader_sentiment(test_text))
