import json
import os

import evaluate
import matplotlib.pyplot as plt
import torch

rouge = evaluate.load("rouge")


def compute_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references)


def eval_rouge(model, val_loader):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in val_loader:
            print(batch)
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            output_ids = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=4,
                early_stopping=True,
            )
            preds = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(preds)

            labels = batch["labels"].clone()
            labels[labels == -100] = model.tokenizer.pad_token_id
            refs = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
            references.extend(refs)

    return rouge.compute(predictions=predictions, references=references)


def plot_training_curves(dataset_name):
    with open(
        f"results/{dataset_name}_training_history.json", "r", encoding="utf-8"
    ) as file_handle:
        history = json.load(file_handle)

    metrics = {
        "train_loss": ("Training Loss", "b-"),
        "rouge1": ("ROUGE-1", "r-"),
        "rouge2": ("ROUGE-2", "g-"),
        "rougeL": ("ROUGE-L", "m-"),
    }

    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, (metric, (title, style)) in enumerate(metrics.items()):
        axis = axes[i // 2, i % 2]
        axis.plot(history["epochs"], history[metric], style)
        axis.set_title(title)
        axis.grid(True)

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(
        f"results/plots/{dataset_name}_training_curves.png", bbox_inches="tight"
    )

    print(
        f"📊 Training curves saved to results/plots/{dataset_name}_training_curves.png"
    )
