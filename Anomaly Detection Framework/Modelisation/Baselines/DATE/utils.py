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
def corrupt_input_random(masked_ids, mask_token_id, vocab_size):
    """
    Random generator (MEILLEUR selon Table 1 du papier)
    Sample uniformément depuis le vocabulaire
    """
    corrupted = masked_ids.clone()
    mask_positions = (masked_ids == mask_token_id)
    
    # Sample aléatoire depuis vocab (excluant les tokens spéciaux)
    n_to_replace = mask_positions.sum().item()
    if n_to_replace > 0:
        # Sample depuis [5, vocab_size) pour éviter [PAD], [UNK], [CLS], [SEP], [MASK]
        random_tokens = torch.randint(
            5, vocab_size, 
            (n_to_replace,), 
            device=masked_ids.device
        )
        corrupted[mask_positions] = random_tokens
    
    return corrupted


@torch.no_grad()
def corrupt_input(generator, masked_ids, attention_mask, mask_token_id):
    """
    Use generator to sample plausible replacements
    (Moins performant que random selon le papier, mais gardé pour compatibilité)
    """
    logits = generator(masked_ids, attention_mask)
    
    # Clamp pour éviter les NaN/Inf
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    probs = torch.softmax(logits, dim=-1)
    
    # Sample from generator distribution
    sampled = torch.multinomial(
        probs.view(-1, probs.size(-1)), 1
    ).view(masked_ids.shape)

    # Replace only [MASK] tokens
    corrupted = masked_ids.clone()
    mask_positions = (masked_ids == mask_token_id)
    corrupted[mask_positions] = sampled[mask_positions]
    
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
    if valid.sum() > 0:
        loss_rtd = bce(
            rtd_logits[valid].float(),
            rtd_labels[valid].float()
        ).mean()
    else:
        loss_rtd = torch.tensor(0.0, device=rtd_logits.device)

    # RMD loss (discriminator, sequence-level)
    loss_rmd = ce(rmd_logits, rmd_labels)

    return mu * loss_rmd + loss_mlm + lambda_rtd * loss_rtd


def apply_mask_safe(input_ids, mask, tokenizer):
    """
    Apply mask pattern while preserving special tokens
    
    IMPORTANT: Ne jamais modifier le mask original !
    On applique le mask mais on skip les tokens spéciaux
    """
    batch_size, seq_len = input_ids.shape
    masked = input_ids.clone()
    
    # Expand mask to batch size (sans modifier l'original)
    mask_expanded = mask.unsqueeze(0).expand(batch_size, -1).clone()
    
    # Create forbidden mask (tokens à ne PAS masquer)
    forbidden = torch.zeros_like(input_ids, dtype=torch.bool)
    forbidden |= (input_ids == tokenizer.cls_token_id)
    forbidden |= (input_ids == tokenizer.sep_token_id)
    forbidden |= (input_ids == tokenizer.pad_token_id)
    
    # Apply mask only where allowed
    final_mask = (mask_expanded == 1) & (~forbidden)
    masked[final_mask] = tokenizer.mask_token_id
    
    return masked, final_mask  # Retourner aussi le masque appliqué


@torch.no_grad()
def date_anomaly_score(
    discriminator,
    dataloader,
    tokenizer,
    device="cuda"
):
    """
    Compute anomaly scores for test data using PL_RTD score
    
    PL_RTD = moyenne de P(token is original) sur tous les tokens valides
    
    Returns:
        scores: np.array, shape (N,) - HIGHER = more normal
        labels: np.array, shape (N,)
    """
    discriminator.eval()

    all_scores = []
    all_labels = []

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward discriminator sur input ORIGINAL (non corrompu)
        rtd_logits, _ = discriminator(input_ids, attention_mask)

        # P(token is replaced) via sigmoid
        p_replaced = torch.sigmoid(rtd_logits)

        # P(token is original) = 1 - P(replaced)
        p_original = 1.0 - p_replaced

        # Masque des tokens valides (selon équation 7 du papier)
        # On EXCLUT : PAD et CLS
        # On GARDE : SEP et tous les autres tokens
        valid = (attention_mask == 1) & \
                (input_ids != tokenizer.cls_token_id) & \
                (input_ids != tokenizer.pad_token_id)
        
        # PL_RTD score = average P(original) over valid tokens
        # Équation (7) du papier
        seq_scores = (
            (p_original * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        )

        all_scores.extend(seq_scores.cpu().numpy())
        all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    
    return scores, labels