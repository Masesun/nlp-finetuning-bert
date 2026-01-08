from transformers import AutoModelForSequenceClassification


def get_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
):
    """
    Initializes a pretrained transformer model for sequence classification.

    Args:
        model_name: Name of the pretrained model.
        num_labels: Number of output classes.

    Returns:
        model: Hugging Face model ready for fine-tuning.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    return model
