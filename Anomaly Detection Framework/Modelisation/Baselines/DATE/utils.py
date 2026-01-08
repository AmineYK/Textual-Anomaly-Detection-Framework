import torch
import numpy as np


def generate_mask_patterns(K, seq_len, ratio):
    """Generate K fixed mask patterns"""
    n = int(seq_len * ratio)
    masks = []
    for _ in range(K):
        idx = torch.randperm(seq_len)[:n]
        m = torch.zeros(seq_len, dtype=torch.long)
        m[idx] = 1
        masks.append(m)
    return torch.stack(masks)



@torch.no_grad()
def corrupt_input(generator, masked_ids, attention_mask, mask_token_id):
    """Use generator to sample plausible replacements"""
    logits = generator(masked_ids, attention_mask)
    probs = torch.softmax(logits, dim=-1)

    # Sample from generator distribution
    sampled = torch.multinomial(
        probs.view(-1, probs.size(-1)), 1
    ).view(masked_ids.shape)

    # Replace only [MASK] tokens
    corrupted = masked_ids.clone()
    corrupted[masked_ids == mask_token_id] = sampled[masked_ids == mask_token_id]
    return corrupted


def date_loss(
    rtd_logits, rmd_logits,
    rtd_labels, rmd_labels,
    mlm_logits, mlm_labels,
    mu=100.0, lambda_rtd=50.0
):
    """DATE combined loss"""
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    # MLM loss (generator)
    loss_mlm = ce(
        mlm_logits.view(-1, mlm_logits.size(-1)),
        mlm_labels.view(-1)
    )

    # RTD loss (discriminator, token-level)
    valid = rtd_labels != -100
    loss_rtd = bce(
        rtd_logits[valid].float(),
        rtd_labels[valid].float()
    ).mean()

    # RMD loss (discriminator, sequence-level)
    loss_rmd = ce(rmd_logits, rmd_labels)

    return mu * loss_rmd + loss_mlm + lambda_rtd * loss_rtd


def apply_mask_safe(input_ids, mask, tokenizer):
    """Apply mask pattern while preserving special tokens"""
    masked = input_ids.clone()

    # Don't mask special tokens
    forbidden = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id
    }

    for tok in forbidden:
        mask = mask & (input_ids != tok)

    masked[mask == 1] = tokenizer.mask_token_id
    return masked


@torch.no_grad()
def date_anomaly_score(
    discriminator,
    dataloader,
    tokenizer,
    device="cuda"
):
    """
    Compute anomaly scores for test data
    
    Returns:
        scores: np.array, shape (N,)
        labels: np.array, shape (N,)
    """
    discriminator.eval()

    all_scores = []
    all_labels = []

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward discriminator
        rtd_logits, _ = discriminator(input_ids, attention_mask)

        # Sigmoid → P(token is replaced)
        p_replaced = torch.sigmoid(rtd_logits)

        # P(token is original)
        p_original = 1.0 - p_replaced

        # Ignore special tokens (CLS, SEP, PAD)
        valid = (attention_mask == 1) & \
                (input_ids != tokenizer.cls_token_id) & \
                (input_ids != tokenizer.sep_token_id) & \
                (input_ids != tokenizer.pad_token_id)
        
        # Score = average P(original) over valid tokens
        seq_scores = (
            (p_original * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1)
        )

        all_scores.extend(seq_scores.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_scores), np.array(all_labels)

    
