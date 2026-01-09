from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from src.dataset import load_and_tokenize_dataset
from src.train import compute_metrics
import json, os

MODEL_NAME = "bert-base-uncased"

tokenized_dataset, tokenizer = load_and_tokenize_dataset()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

training_args = TrainingArguments(
    output_dir="results",
    per_device_eval_batch_size=16,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate()

os.makedirs("results", exist_ok=True)
with open("results/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)

