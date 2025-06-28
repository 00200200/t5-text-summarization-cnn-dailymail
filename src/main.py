import json
import os
import sys

from datasets import load_dataset

from src.data.dataio import get_data_loaders
from src.model.model import SummarizerModel
from src.train.train import train_model
from src.utils.utils import plot_training_curves

DATASET_CONFIGS = {
    "xsum": ("EdinburghNLP/xsum", None, "document", "summary"),
    "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "article", "highlights"),
}


def generate_examples(model, dataset_name, num_examples=3):
    path, version, text_field, summary_field = DATASET_CONFIGS[dataset_name]
    dataset = (
        load_dataset(path, version, split="test")
        if version
        else load_dataset(path, split="test")
    )

    examples = []
    for i in range(num_examples):
        text = dataset[i][text_field]
        examples.append(
            {
                "text": text,
                "reference": dataset[i][summary_field],
                "prediction": model.generate_summary(text),
            }
        )
    return examples


def train_dataset(
    dataset_name, train_size=30000, val_size=1000, batch_size=16, epochs=5
):
    print(f"\n🚀 Training on {dataset_name.upper()}")
    print(f"📊 {train_size} train samples, batch_size={batch_size}, epochs={epochs}")

    model = SummarizerModel()
    train_loader, val_loader = get_data_loaders(
        dataset_name, batch_size, train_size, val_size, model.tokenizer
    )
    model, history = train_model(model, train_loader, val_loader, epochs, dataset_name)

    examples = generate_examples(model, dataset_name)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{dataset_name}_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    plot_training_curves(dataset_name)


def main():
    print("📝 T5 Text Summarizer Training")
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if dataset in ["xsum", "cnn"]:
        dataset_name = "xsum" if dataset == "xsum" else "cnn_dailymail"
        train_dataset(dataset_name)
    else:
        print("🎯 Training on both datasets")
        train_dataset("xsum")
        train_dataset("cnn_dailymail")

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
