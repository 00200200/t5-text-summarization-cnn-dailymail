import json
import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler

from src.utils.utils import eval_rouge


def train_model(model, train_loader, val_loader, epochs=5, dataset_name="xsum"):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )
    history = {"epochs": [], "train_loss": [], "rouge1": [], "rouge2": [], "rougeL": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(model.device) for k, v in batch.items()}

            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.3f}"})

        print("Evaluating...")
        avg_loss = total_loss / len(train_loader)
        progress_bar.set_description(f"Evaluating epoch {epoch+1}")
        rouge_scores = eval_rouge(model, val_loader)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["rouge1"].append(rouge_scores["rouge1"])
        history["rouge2"].append(rouge_scores["rouge2"])
        history["rougeL"].append(rouge_scores["rougeL"])

        print(
            f"Epoch {epoch+1}: Loss {avg_loss:.3f}, ROUGE-1{rouge_scores['rouge1']:.3f}"
        )

        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/{dataset_name}_epoch_{epoch+1}.pth")

    # Save history
    os.makedirs("results", exist_ok=True)
    with open(
        f"results/{dataset_name}_training_history.json", "w", encoding="utf-8"
    ) as file_handle:
        json.dump(history, file_handle, indent=2)

    return model, history
