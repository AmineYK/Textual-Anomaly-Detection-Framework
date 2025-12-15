import torch
import torch.nn as nn
from transformers import ElectraModel

class DATE(nn.Module):
    def __init__(self,
                 model_name="google/electra-base-discriminator",
                 n_patterns=4):
        super().__init__()

        self.encoder = ElectraModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # TRD / RTD head (token-level)
        self.token_head = nn.Linear(hidden_size, 2)

        # RMD head (sequence-level)
        self.sequence_head = nn.Linear(hidden_size, n_patterns)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                input_ids,
                attention_mask,
                token_labels=None,
                sequence_labels=None):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden = outputs.last_hidden_state          # (B, L, H)
        cls = hidden[:, 0]                          # (B, H)

        token_logits = self.token_head(hidden)      # (B, L, 2)
        seq_logits = self.sequence_head(cls)        # (B, P)

        loss = None
        token_loss = None
        seq_loss = None

        if token_labels is not None:
            token_loss = self.ce_loss(
                token_logits.view(-1, 2),
                token_labels.view(-1)
            )

        if sequence_labels is not None:
            seq_loss = self.ce_loss(seq_logits, sequence_labels)

        if token_loss is not None and seq_loss is not None:
            loss = token_loss + seq_loss

        return {
            "loss": loss,
            "token_loss": token_loss,
            "seq_loss": seq_loss
        }

import random

def electra_corruption(input_ids, tokenizer, corruption_rate=0.15):
    corrupted = input_ids.clone()
    labels = torch.zeros_like(input_ids)

    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    for i in range(input_ids.size(0)):
        for j in range(input_ids.size(1)):
            if input_ids[i, j].item() in special_ids:
                continue

            if random.random() < corruption_rate:
                corrupted[i, j] = random.randint(0, vocab_size - 1)
                labels[i, j] = 1

    return corrupted, labels

def train_date(model, dataloader, tokenizer, optimizer, device, n_patterns=4):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        corrupted, token_labels = electra_corruption(input_ids, tokenizer)
        corrupted = corrupted.to(device)
        token_labels = token_labels.to(device)

        # RMD label (pattern id)
        seq_labels = torch.randint(
            0, n_patterns, (input_ids.size(0),),
            device=device
        )

        out = model(
            corrupted,
            attention_mask,
            token_labels,
            seq_labels
        )

        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def score_samples(model, dataloader, tokenizer, device):
    model.eval()
    scores = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        corrupted, token_labels = electra_corruption(input_ids, tokenizer)
        corrupted = corrupted.to(device)
        token_labels = token_labels.to(device)

        out = model(
            corrupted,
            attention_mask,
            token_labels,
            sequence_labels=None
        )

        scores.append(out["token_loss"].cpu())

    return torch.stack(scores).numpy()
