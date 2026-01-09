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
# BATCH INFERENCE
# =========================

import pandas as pd
from tqdm import tqdm

@torch.no_grad()
def predict_batch(
    texts,
    tokenizer,
    model,
    batch_size=8
):
    results = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        for j, text in enumerate(batch_texts):
            pred_id = preds[j].item()
            results.append({
                "text": text,
                "label": model.config.id2label[pred_id],
                "confidence": round(probs[j, pred_id].item(), 4)
            })

    return pd.DataFrame(results)

def run_batch_inference(
    input_csv: str,
    output_csv: str,
    text_column: str = "text",
    batch_size: int = 8
):
    tokenizer, model = load_model()

    df = pd.read_csv(input_csv)
    texts = df[text_column].astype(str).tolist()

    preds_df = predict_batch(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size
    )

    preds_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

# =========================
# SANITY CHECK
# =========================

if __name__ == "__main__":
    tokenizer, model = load_model()

    sample_text = "Psilocybin induces rapid structural plasticity in the prefrontal cortex."

    result = predict(sample_text, tokenizer, model)
    print(result)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=False)
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    if args.input_csv:
        run_batch_inference(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            batch_size=args.batch_size
        )
    else:
        tokenizer, model = load_model()
        sample_text = "Psilocybin induces rapid structural plasticity in the prefrontal cortex."
        print(predict(sample_text, tokenizer, model))


