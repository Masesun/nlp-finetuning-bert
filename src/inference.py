import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================

MODEL_PATH = "/content/drive/MyDrive/bert-imdb"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOADERS
# =========================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.config.id2label = {
        0: "NEGATIVE",
        1: "POSITIVE"
    }
    model.config.label2id = {
        "NEGATIVE": 0,
        "POSITIVE": 1
    }
    model.to(DEVICE)
    model.eval()

    return tokenizer, model

# =========================
# INFERENCE
# =========================

@torch.no_grad()
def predict(text: str, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    outputs = model(**inputs)
    logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_id].item()

    label = model.config.id2label[pred_id]

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            model.config.id2label[i]: round(p.item(), 4)
            for i, p in enumerate(probs[0])
        }
    }

# =========================
# SANITY CHECK
# =========================

if __name__ == "__main__":
    tokenizer, model = load_model()

    sample_text = "Psilocybin induces rapid structural plasticity in the prefrontal cortex."

    result = predict(sample_text, tokenizer, model)
    print(result)
