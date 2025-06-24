import torch
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration


class SummarizerModel(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        self.model.to(self.device)

    def generate_summary(self, text):
        input_ids = self.tokenizer(
            f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True
        ).input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids, max_length=150, num_beams=4, early_stopping=True
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
