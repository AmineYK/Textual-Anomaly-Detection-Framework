# data/mask_generator.py
import torch

class MaskGenerator:
    def __init__(self, num_masks: int, seq_len: int, mask_ratio: float = 0.5, seed: int = 42):
        self.num_masks = num_masks
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        torch.manual_seed(seed)
        self.masks = self._generate_masks()

    def _generate_masks(self):
        masks = torch.zeros(self.num_masks, self.seq_len, dtype=torch.bool)
        num_masked = int(self.seq_len * self.mask_ratio)

        for k in range(self.num_masks):
            idx = torch.randperm(self.seq_len)[:num_masked]
            masks[k, idx] = True

        return masks

    def sample(self, batch_size: int):
        mask_ids = torch.randint(0, self.num_masks, (batch_size,))
        return self.masks[mask_ids], mask_ids
    

# model/discriminator.py
import torch
import torch.nn as nn
from transformers import ElectraForPreTraining

class DATEDiscriminator(nn.Module):
    def __init__(self, electra_name: str, num_masks: int):
        super().__init__()
        self.electra = ElectraForPreTraining.from_pretrained(electra_name)
        hidden = self.electra.config.hidden_size

        self.rmd_head = nn.Linear(hidden, num_masks)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        rtd_logits = outputs.logits
        cls_embedding = outputs.hidden_states[-1][:, 0]
        rmd_logits = self.rmd_head(cls_embedding)

        return rtd_logits, rmd_logits
    

# model/date_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraForMaskedLM

class DATEModel(nn.Module):
    def __init__(
        self,
        electra_generator: str,
        electra_discriminator: str,
        num_masks: int,
        lambda_rtd: float = 50.0,
        mu_rmd: float = 1.0
    ):
        super().__init__()
        self.generator = ElectraForMaskedLM.from_pretrained(electra_generator)
        self.discriminator = DATEDiscriminator(electra_discriminator, num_masks)

        self.lambda_rtd = lambda_rtd
        self.mu_rmd = mu_rmd

        self.loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_rtd = nn.BCEWithLogitsLoss()
        self.loss_rmd = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        masked_input_ids,
        mlm_labels,
        rtd_labels,
        rmd_labels
    ):
        # Generator
        gen_outputs = self.generator(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels
        )
        loss_mlm = gen_outputs.loss

        with torch.no_grad():
            sampled_tokens = torch.argmax(gen_outputs.logits, dim=-1)

        replaced_input_ids = masked_input_ids.clone()
        replaced_input_ids[mlm_labels != -100] = sampled_tokens[mlm_labels != -100]

        # Discriminator
        rtd_logits, rmd_logits = self.discriminator(
            replaced_input_ids,
            attention_mask
        )

        loss_rtd = self.loss_rtd(
            rtd_logits.view(-1),
            rtd_labels.float().view(-1)
        )

        loss_rmd = self.loss_rmd(rmd_logits, rmd_labels)

        loss = loss_mlm + self.lambda_rtd * loss_rtd + self.mu_rmd * loss_rmd

        return {
            "loss": loss,
            "loss_mlm": loss_mlm.detach(),
            "loss_rtd": loss_rtd.detach(),
            "loss_rmd": loss_rmd.detach(),
        }
    

# inference/anomaly_score.py
import torch

@torch.no_grad()
def compute_anomaly_score(discriminator, input_ids, attention_mask):
    rtd_logits, _ = discriminator(input_ids, attention_mask)
    p_original = torch.sigmoid(rtd_logits)
    return p_original.mean(dim=1)



