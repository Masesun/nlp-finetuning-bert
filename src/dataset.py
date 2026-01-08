from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenize_dataset(
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
):
    """
    Loads the IMDB dataset and tokenizes text using a pretrained tokenizer.

    Args:
        model_name: Name of the pretrained model tokenizer.
        max_length: Maximum sequence length.

    Returns:
        tokenized_dataset: Hugging Face DatasetDict ready for training.
        tokenizer: Initialized tokenizer.
    """

    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    return tokenized_dataset, tokenizer
