import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

DATASET_CONFIGS = {
    "xsum": ("EdinburghNLP/xsum", None, "document", "summary"),
    "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "article", "highlights"),
}


class SummarizerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, text_field, summary_field):
        self.examples = dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.summary_field = summary_field

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx][self.text_field]
        summary = self.examples[idx][self.summary_field]

        inputs = self.tokenizer(
            f"summarize: {text}",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            summary,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.squeeze(),
        }


def get_data_loaders(dataset_name, batch_size, train_size, val_size, tokenizer):
    path, version, text_field, summary_field = DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(path, version) if version else load_dataset(path)

    train_dataset = SummarizerDataset(
        dataset["train"].select(range(train_size)), tokenizer, text_field, summary_field
    )

    val_dataset = SummarizerDataset(
        dataset["validation"].select(range(val_size)),
        tokenizer,
        text_field,
        summary_field,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
