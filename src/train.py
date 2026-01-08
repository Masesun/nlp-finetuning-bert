from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import load_and_tokenize_dataset
from src.model import get_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def train():
    tokenized_dataset, tokenizer = load_and_tokenize_dataset()
    model = get_model()

    training_args = TrainingArguments(
        output_dir="results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16 = True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    train()
