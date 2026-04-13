import math
import torch
from torch import nn,  Tensor
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from flow_matching.solver import ODESolver
import time
from torch.utils.data import TensorDataset, DataLoader
import Modelisation.evaluation as ev


class BatchedVelocityWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        # t est scalaire → on l'étend au batch
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Zero-init adaLN (comme l'original)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)


    def forward(self, x, t, return_attention=False):
        """
        x : (B, D)    → sentence level
        x : (B, T, D) → token level
        c : (B, D)    → timestep embedding (toujours)
        """
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t).chunk(6, dim=-1)  # (B, D) chacun

        # --- Gestion sentence vs token ---
        # is_sentence = (x.dim() == 2)
        # if is_sentence:
        #     x = x.unsqueeze(1)  # (B, 1, D) — séquence de longueur 1

        # Maintenant x est toujours (B, T, D)
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)  # (B, T, D)
        attn_out, attn_weights = self.attn(x_mod, x_mod, x_mod,
                              need_weights=True,
                              average_attn_weights=True)

        # gate : (B, D) → (B, 1, D) pour broadcaster sur T
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        if return_attention:
            return x, attn_weights  # (B, T, T)

        return x, None


class FinalLayer(nn.Module):
    """Couche finale avec adaLN — comme l'original"""
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Zero-init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias,   0)

    def forward(self, x, c):
        """
        x : (B, D) ou (B, T, D)
        c : (B, D)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

def modulate(x, shift, scale):
    if x.dim() == 2:
        # Sentence level : (B, D)
        return x * (1 + scale) + shift
    else:
        # Token level : (B, T, D)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FlowDiTToken(nn.Module):
    """
    DiT générique : sentence level (B, D) ou token level (B, T, D)
    """
    def __init__(
        self,
        latent_dim=768,        # dimension par token SBERT
        hidden_dim=256,
        depth=4,
        n_heads=4,
        mlp_ratio=4.0,
        freq_embed_size=256,
    ):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_dim, freq_embed_size)

        # Blocs DiT
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ✅ AJOUT : query appris pour attention pooling
        self.attn_pool_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )
        self.attn_pool = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True
        )

        # Couche finale avec adaLN
        self.final_layer = FinalLayer(hidden_dim, latent_dim)
        self._init_weights()

        # Init weights
        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # ✅ Ré-appliquer le zero-init APRÈS _basic_init
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,   0)

        # ✅ Zero-init FinalLayer APRÈS _basic_init
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias,   0)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x_tokens, t, attention_mask=None, return_attention=False):
        """
        x_tokens       : (B, T, 768)  ← vrais token embeddings SBERT
        t              : (B,)
        attention_mask : (B, T)       ← 1=vrai token, 0=padding (optionnel)

        Retourne :
            v_sentence : (B, 768)     ← velocity field au niveau phrase
            v_tokens   : (B, T, 768)  ← velocity field par token
        """

        B, T, _ = x_tokens.shape
        # Projection tokens → hidden_dim
        h = self.input_proj(x_tokens)       # (B, T, hidden_dim)

        # Timestep embedding
        c = self.t_embedder(t)              # (B, hidden_dim)

        # # DiT blocks
        # for block in self.blocks:
        #     h = block(h, c)                 # (B, T, hidden_dim)
        all_attentions = []
        for block in self.blocks:
            h, attn_w = block(h, c, return_attention=return_attention)
            if return_attention and attn_w is not None:
                all_attentions.append(attn_w)  # (B, T, T)


        # ✅ Attention pooling : query appris interroge tous les tokens
        query = self.attn_pool_query.expand(B, 1, -1)  # (B, 1, hidden_dim)


        # # FinalLayer → velocity field par token
        # v_tokens = self.final_layer(h, c)   # (B, T, 768)

        # # ✅ Mean pooling avec masque si padding
        # if attention_mask is not None:
        #     # Ne moyenne que sur les vrais tokens
        #     mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        #     v_sentence = (v_tokens * mask).sum(dim=1) / \
        #                   mask.sum(dim=1).clamp(min=1e-8) # (B, 768)
        # else:
        #     v_sentence = v_tokens.mean(dim=1)            # (B, 768)

        # if return_attention:
        #     # Stack : (n_blocks, B, T, T)
        #     all_attentions = torch.stack(all_attentions, dim=0)
        #     return v_sentence, v_tokens, all_attentions

        # return v_sentence, v_tokens, None

        # Masque pour ignorer PAD dans l'attention pooling
        key_padding_mask = None
        if attention_mask is not None:
            # MultiheadAttention attend True = ignorer
            key_padding_mask = (attention_mask == 0)   # (B, T) bool

        pool_out, pool_attn = self.attn_pool(
            query,                                      # (B, 1, hidden_dim)
            h,                                          # (B, T, hidden_dim)
            h,                                          # (B, T, hidden_dim)
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        # pool_out  : (B, 1, hidden_dim)
        # pool_attn : (B, 1, T) ← poids d'attention sur les tokens

        h_sentence = pool_out.squeeze(1)               # (B, hidden_dim)

        # ✅ FinalLayer sur la représentation sentence
        v_sentence = self.final_layer(h_sentence, c)   # (B, 768)

        # ✅ v_tokens pour insights XAI — projection directe
        v_tokens = self.final_layer(h, c)              # (B, T, 768)

        if return_attention:
            all_attentions = torch.stack(all_attentions, dim=0)  # (n_blocks, B, T, T)
            return v_sentence, v_tokens, all_attentions, pool_attn

        return v_sentence, v_tokens, None


class FlowMatchingTransformersToken(nn.Module):

    def __init__(self, model, source, target, config, noise_is_target, rectified):
        super().__init__()
        self.model = model
        self.source = source
        self.target = target
        self.noise_is_target = noise_is_target
        if self.noise_is_target:
            self.device = self.source.device
            # self.centroid = Tensor(self.source.mean(dim=0)).to(self.device)
            # self.var = Tensor(self.source.var(dim=0)).mean().to(self.device)
            # ✅ MODIF : centroid sur [CLS] uniquement (index 0)
            # cls_tokens = self.source[:, 0, :]               # (N_samples, D)
            # self.centroid = cls_tokens.mean(dim=0)          # (D,)
            # self.var = cls_tokens.var(dim=0).mean()


            source_tokens = self.source           # (N, T, 768)
            # Mean pooling sur tous les tokens non-PAD
            # On suppose que source est déjà masqué
            self.centroid = source_tokens.mean(dim=1).mean(dim=0)  # (768,)
            self.var      = source_tokens.mean(dim=1).var(dim=0).mean()

            # self.centroid = torch.ones((self.source.shape[1]), device=self.device)*2
        else:
            self.device = self.target.device
            # ✅ MODIF : centroid sur [CLS] uniquement (index 0)
            cls_tokens = self.target[:, 0, :]               # (N_samples, D)
            self.centroid = cls_tokens.mean(dim=0)          # (D,)
            self.var = cls_tokens.var(dim=0).mean()
            # self.centroid = Tensor(self.target.mean(dim=0)).to(self.device)
            # self.var = Tensor(self.target.var(dim=0)).mean().to(self.device)
            # self.centroid = torch.ones((self.target.shape[1]), device=self.device)*2
        self.rectified = rectified
        self.config = config

        self.log_r = nn.Parameter(torch.tensor(0.0).to(self.device))

        # dist_train = torch.norm(self.source - self.centroid, dim=1)
        # r_init = torch.quantile(dist_train, 0.95)
        # self.log_r = nn.Parameter(torch.log(r_init))
        # print(self.log_r)


        # self.log_margin = nn.Parameter(torch.tensor(-1.0).to(self.device))
        self.log_margin = nn.Parameter(torch.tensor(0.0).to(self.device))
        # self.margin = 0.1


        # because the source and target distirbution are inlier and inlier_masked
        self.is_masking_task = type(self.config['source']) != str and type(self.config['target']) != str

    @property
    def margin(self):
        return torch.exp(self.log_margin) + 0.05

    @property
    def r_in(self):
        return torch.exp(self.log_r)

    @property
    def r_out(self):
        return self.r_in + self.margin

    def get_lr_schedule(self,epoch, warmup_epochs, total_epochs, lr):

        # return lr
        # linear increase
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / (warmup_epochs + 1)
        # cosinus decrease
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return (lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))).item()


    def get_lambda_kl(self, epoch, lambda_max, kl_warmup_epochs):
        return lambda_max * min(1.0, epoch / kl_warmup_epochs)

    def get_attention_maps(self, x_tokens, t_val=0.5, attention_mask=None):
        """
        Retourne les attention maps de tous les blocs
        x_tokens : (B, T, 768)
        → all_attentions : (n_blocks, B, T, T)
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.full((x_tokens.shape[0],), t_val, device=self.device)

            _, _, all_attentions = self.model(
                x_tokens, t,
                attention_mask=attention_mask,
                return_attention=True
            )

        return all_attentions  # (n_blocks, B, T, T)

    def sample_like(self, x_0, type, tail=None):

        if type == 'gaussian':
            return torch.randn_like(x_0)

        if type == 'gaussian-neigh':
            B, T, D = x_0.shape
            sigma = torch.sqrt(self.var * self.config['coef_var'])
            # x_mean = x_0.mean(dim=1, keepdim=True).expand_as(x_0)  # (B, T, 768)
            # return x_mean + sigma * torch.randn_like(x_0)
            if self.centroid.dim == 1:
                self.centroid = self.centroid.unsqueeze(0).unsqueeze(0).repeat(B, T, 1)
            return self.centroid + sigma * torch.randn_like(x_0)

        if type == 'centroid':
            return self.centroid.repeat(x_0.shape[0],1)

        if type =='sphere-noised':
            z = torch.randn(x_0.shape[0], x_0.shape[1])
            z = z / z.norm(dim=1, keepdim=True)
            noise = torch.randn_like(z) * 0.25
            return Tensor(z + noise).to(self.device)

        if type == 'sphere':
            z = torch.randn(x_0.shape[0], x_0.shape[1])
            return Tensor(z / z.norm(dim=1, keepdim=True)).to(self.device)


    # def _euler_integrate_single(self, x_0, mask, N_steps=10, save_all=False):
    #     with torch.no_grad():
    #         x_t = x_0
    #         dt = 1.0 / N_steps

    #         if save_all:
    #             velocities_tokens = []
    #             x_t_inter = [x_t.detach().cpu()]

    #         for i in range(N_steps):
    #             t_val = i * dt
    #             t = torch.full((x_t.shape[0],), t_val, device=self.device)

    #             v_sen, v_tokens, _ = self.model(x_t, t, mask)
    #             x_t = x_t + dt * v_tokens

    #             if save_all:
    #                 velocities_tokens.append(v_tokens.detach().cpu())
    #                 x_t_inter.append(x_t.detach().cpu())

    #         if save_all:
    #             velocities_tokens = torch.stack(velocities_tokens)
    #             x_t_inter = torch.stack(x_t_inter)
    #         else:
    #             velocities_tokens, x_t_inter = None, None

    #     return x_t, velocities_tokens, x_t_inter

    def _euler_integrate_single(self, x_0, mask, N_steps=10, save_all=False):
        with torch.no_grad():

            # ✅ x_0_sentence pour le transport
            mask_exp     = mask.unsqueeze(-1).float()
            x_0_sentence = (x_0 * mask_exp).sum(dim=1) / \
                            mask_exp.sum(dim=1).clamp(min=1e-8)   # (B, 768)

            x_t_sent = x_0_sentence.clone()
            dt = 1.0 / N_steps

            if save_all:
                velocities_tokens = []
                x_t_inter         = [x_t_sent.detach().cpu()]

            for i in range(N_steps):
                t_val = i * dt
                t     = torch.full(
                    (x_t_sent.shape[0],), t_val, device=self.device
                )

                # ✅ x_t_tokens = x_t_sentence + résidu original
                residual   = x_0 - x_0_sentence.unsqueeze(1)
                x_t_tokens = x_t_sent.unsqueeze(1) + residual     # (B, T, 768)

                v_sentence, v_tokens, _ = self.model(x_t_tokens, t, mask)

                # ✅ Intégration au niveau sentence
                x_t_sent = x_t_sent + dt * v_sentence             # (B, 768)

                if save_all:
                    velocities_tokens.append(v_tokens.detach().cpu())
                    x_t_inter.append(x_t_sent.detach().cpu())

            if save_all:
                velocities_tokens = torch.stack(velocities_tokens)
                x_t_inter         = torch.stack(x_t_inter)
            else:
                velocities_tokens, x_t_inter = None, None

        return x_t_sent, velocities_tokens, x_t_inter


    def euler_integrate(self, x_0, mask, N_steps=10, save_all=False, batch_size=64):
        all_x_final = []

        if save_all:
            all_velocities = []
            all_x_inter = []

        for start in range(0, x_0.shape[0], batch_size):
            end = start + batch_size

            x_batch = x_0[start:end].to(self.device)
            mask_batch = mask[start:end].to(self.device)

            x_final, velocities, x_inter = self._euler_integrate_single(
                x_batch, mask_batch, N_steps=N_steps, save_all=save_all
            )

            all_x_final.append(x_final.detach().cpu())

            if save_all:
                all_velocities.append(velocities)
                all_x_inter.append(x_inter)

            del x_batch, mask_batch, x_final, velocities, x_inter
            torch.cuda.empty_cache()

        all_x_final = torch.cat(all_x_final, dim=0)

        if save_all:
            all_velocities = torch.cat(all_velocities, dim=1)
            all_x_inter = torch.cat(all_x_inter, dim=1)
            return all_x_final, all_velocities, all_x_inter

        return all_x_final.to(self.device), None, None

    # def compute_flow_loss(self, x_0, mask_x_0, indices, flow_type='linear', sigma=0.1, warmup_activated=False):


    #     mask = mask_x_0.clone() 

    #     if self.target == 'gaussian-neigh' or self.source == 'gaussian-neigh':
    #         x_1 = self.sample_like(x_0, 'gaussian-neigh')

    #     batch_size = x_1.shape[0]
    #     t = torch.rand(batch_size, device=self.device)

    #     seq_lengths = mask.sum(dim=1).long()  # (B,) — longueur réelle
    #     for b in range(x_0.shape[0]):
    #         mask[b, seq_lengths[b] - 1] = 0  # ← ignore SEP

    #     # j'inverse juste la source et la target
    #     if not self.noise_is_target:
    #         x_0, x_1 = x_1, x_0

    #     if flow_type == 'linear':

    #         t_expanded = t.view(-1, 1, 1)
    #         x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    #         # v_target = x_1 - x_0
    #         v_target = (x_1 - x_0).mean(dim=1)

    #     elif flow_type == 'cfm':
    #         t_expanded = t.view(-1, 1, 1)
    #         mu_t = t_expanded * x_1 + (1 - t_expanded) * x_0

    #         eps = torch.randn_like(x_0)
    #         x_t = mu_t + sigma * eps

    #         v_target = x_1 - x_0

    #     else:
    #         raise ValueError(f"Unknown flow_type: {flow_type}")

    #     # v_pred = self.model(x_t, t)
    #     v_pred, v_tokens, _ = self.model(x_t, t, mask)

    #     mask_expanded = mask.unsqueeze(-1).float()  
    #     loss_fm = (
    #         F.mse_loss(v_tokens, v_target, reduction='none')
    #         * mask_expanded                                   
    #     ).sum() / mask_expanded.sum()   

    #     # <<<<<<<<<<<<<<<<<<<<< LOSS FM >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #     # loss_fm = F.mse_loss(v_pred, v_target)

    #     # loss_fm = torch.tensor(0.0, device=self.device, requires_grad=True)

    #     loss_svdd = torch.tensor(0.0, device=self.device)

    #     if not warmup_activated:

    #         # <<<<<<<<<<<<<<<<<<<<< LOSS SVDD >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         if self.config['lambda_svdd'] > 0:
    #             x_svdd, _, _ = self.euler_integrate(x_0, mask, N_steps=self.config['n_step_euler_integrate'], save_all=False)
    #             # x_mean = x_0.mean(dim=1, keepdim=True).expand_as(x_0)  # (B, T, 768)
    #             dist_sq = torch.sum((x_svdd - self.centroid)**2, dim=1)
    #             # dist_sq = torch.sum((x_svdd.mean(dim=1) - self.centroid)**2, dim=1)
    #             # dist_sq = torch.sum((x_svdd - x_mean)**2, dim=1)

    #             r_sq = self.r_in ** 2
    #             loss_svdd = r_sq + F.relu(dist_sq - r_sq).mean()

    #             loss_total = loss_fm + self.config['lambda_svdd'] * loss_svdd

    #             return loss_total, loss_fm.item(), loss_svdd.item()

    #     else:

    #         loss_total = loss_fm
    #         return loss_total, loss_fm, 0

    def compute_flow_loss(self, x_0, mask_x_0, indices,
                          flow_type='linear', sigma=0.1,
                          warmup_activated=False):

        mask = mask_x_0.clone()

        # ✅ x_0_sentence : mean pooling masqué des tokens
        mask_exp      = mask.unsqueeze(-1).float()              # (B, T, 1)
        x_0_sentence  = (x_0 * mask_exp).sum(dim=1) / \
                         mask_exp.sum(dim=1).clamp(min=1e-8)    # (B, 768)

        # ✅ x_1 : gaussienne compacte autour de mu — niveau sentence
        B = x_0.shape[0]
        sigma_val = torch.sqrt(self.var * self.config['coef_var'])
        x_1_sentence = self.centroid + sigma_val * \
                        torch.randn(B, self.centroid.shape[0],
                                    device=self.device)          # (B, 768)

        # Ignore SEP token
        seq_lengths = mask.sum(dim=1).long()
        for b in range(B):
            mask[b, seq_lengths[b] - 1] = 0

        t = torch.rand(B, device=self.device)

        if flow_type == 'linear':
            t_exp      = t.view(-1, 1)
            x_t_sent   = t_exp * x_1_sentence + \
                         (1 - t_exp) * x_0_sentence              # (B, 768)
            v_target   = x_1_sentence - x_0_sentence             # (B, 768)

        # ✅ Construire x_t_tokens : x_t_sentence + résidu token
        residual   = x_0 - x_0_sentence.unsqueeze(1)             # (B, T, 768)
        x_t_tokens = x_t_sent.unsqueeze(1) + residual            # (B, T, 768)

        # Forward DiT
        v_sentence, v_tokens, _ = self.model(x_t_tokens, t, mask)

        # ✅ Loss FM au niveau sentence uniquement
        loss_fm = F.mse_loss(v_sentence, v_target)

        if not warmup_activated:

            if self.config['lambda_svdd'] > 0:

                # SVDD au niveau sentence
                t_zeros    = torch.zeros(B, device=self.device)
                x_0_res    = x_0 - x_0_sentence.unsqueeze(1)     # résidu
                x_t_zeros  = x_0_sentence.unsqueeze(1) + x_0_res # (B, T, 768)

                v_svdd, _, _ = self.model(x_t_zeros, t_zeros, mask)
                phi_1        = x_0_sentence + v_svdd              # (B, 768)

                dist_sq = torch.sum(
                    (phi_1 - self.centroid)**2, dim=1
                )                                                  # (B,)
                r_sq      = self.r_in ** 2
                loss_svdd = r_sq + F.relu(dist_sq - r_sq).mean()

                loss_total = loss_fm + \
                             self.config['lambda_svdd'] * loss_svdd
                return loss_total, loss_fm.item(), loss_svdd.item()

        loss_total = loss_fm
        return loss_total, loss_fm.item(), 0


    def train_epoch(self, dataloader, optimizer, warmup_activated):

        self.model.train()
        self.optimizer = optimizer

        total_loss_liste = []
        loss_fm_liste = []
        loss_svdd_liste = []

        # for x_0, y, indices in dataloader:
        for x_0, mask_x_0, indices in dataloader:

            x_0 = x_0.to(self.device)
            loss_total, *anythingelse = self.compute_flow_loss(
            # loss_total, loss_fm, loss_svdd, loss_push, margin_value, r_in_value = self.compute_flow_loss(
                x_0,
                mask_x_0,
                indices,
                flow_type=self.config['flow_type'],
                sigma=self.config['sigma'],
                warmup_activated=warmup_activated
            )

            self.optimizer.zero_grad()
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [self.log_r, self.log_margin],
                self.config['grad_clip']
            )

            self.optimizer.step()

            total_loss_liste.append(loss_total.item())
            loss_fm_liste.append(anythingelse[0])
            loss_svdd_liste.append(anythingelse[1])

        return np.mean(total_loss_liste), np.mean(loss_fm_liste), np.mean(loss_svdd_liste)


    def train(self, attentions_mask, verbose=True):

        optimizer = AdamW(
            # self.model.parameters(),
            list(self.model.parameters()) + [self.log_r] + [self.log_margin],
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            foreach=False
        )

        if self.noise_is_target:
            X_inlier = self.source
        else:
            X_inlier = self.target

        # X_inlier_dl = DataLoader(TensorDataset(X_inlier, self.config['y_train'], torch.arange(len(X_inlier))), batch_size=self.config['batch_size'], shuffle=True)
        X_inlier_dl = DataLoader(TensorDataset(X_inlier, attentions_mask, torch.arange(len(X_inlier))), batch_size=self.config['batch_size'], shuffle=True)


        total_loss_liste = []
        loss_fm_liste = []
        loss_svdd_liste = []


        for epoch in range(self.config['epochs']):
            warmup_activated = epoch <= self.config['warmup_epochs']

            if self.config['warmup_epochs'] == 0 and epoch == self.config['warmup_epochs']:
                self.log_r.data.fill_(0.0)
                print(f"Initialization r_in : {self.r_in}")

            elif epoch == self.config['warmup_epochs']:
                self.model.eval()
                with torch.no_grad():

                    all_dists = []
                    for x_batch, _ in X_inlier_dl:
                        x_batch = x_batch.to(self.device)
                        t_zeros = torch.zeros(x_batch.shape[0]).to(self.device)
                        v = self.model(x_batch, t_zeros)
                        phi_1 = x_batch + v
                        dist = torch.norm(phi_1 - self.centroid, dim=1)
                        all_dists.append(dist)

                    all_dists = torch.cat(all_dists)
                    r_init = torch.quantile(all_dists, 0.90)
                    self.log_r.data.fill_(torch.log(r_init).item())
                print(f"FM Warmup is finished after .... initialization r_in : {self.r_in}")
                self.model.train()

            lr = self.get_lr_schedule(epoch, self.config['lr_epochs'], self.config['epochs'], self.config['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_total, *anythingelse = self.train_epoch(
            # loss_total, loss_fm, loss_svdd, loss_push, margin_value, r_in_value = self.train_epoch(
                X_inlier_dl,
                optimizer,
                warmup_activated,
            )
            total_loss_liste.append(loss_total.item())
            loss_fm_liste.append(anythingelse[0])
            loss_svdd_liste.append(anythingelse[1])

            if epoch % (self.config['epochs'] // 3) == 0 and verbose:
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                print(f"Train Loss: {loss_total:.4f}, LR: {lr:.6f}")
                print(self.r_in)


        if self.rectified is None:
            return total_loss_liste, loss_fm_liste, loss_svdd_liste
        else:
            print(f"\nRectification Pass starting.... for {self.rectified} iterations")

            for iteration in range(self.rectified):
                l = []
                for x_0, in X_inlier_dl:

                    traj, v_rectified = self.generate_rectified_targets(self.model, x_0)
                    loss = self.rectified_flow_iterative_loss(self.model, traj, v_rectified)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    l.append(loss.item())

                print(f"Rectification iteration {iteration+1}, loss = {np.mean(l):.6f}")

            return liste_loss

    @torch.no_grad()
    def generate_rectified_targets(self, model, x0, steps=50):
        B = x0.shape[0]
        x = x0.clone()
        dt = 1.0 / steps

        traj = []
        for i in range(steps):
            t = torch.full((B,), i * dt, device=x.device)
            x = x + model(x, t) * dt
            traj.append(x.clone())

        xT = x
        v_rectified = (xT - x0)

        return traj, v_rectified

    def rectified_flow_iterative_loss(self, model, traj, v_rectified):
        losses = []
        steps = len(traj)

        for i, xt in enumerate(traj):
            B = xt.shape[0]
            t = torch.full((B,), i / steps, device=xt.device)
            v_pred = model(xt, t)
            losses.append(((v_pred - v_rectified) ** 2).mean())

        return sum(losses) / len(losses)

    def forward_flow(self, x_0, solver_type='midpoint', n_steps=10):

        self.model.eval()
        wrapped_model = BatchedVelocityWrapper(self.model)
        solver = ODESolver(velocity_model=wrapped_model)

        time_steps = torch.linspace(0.0, 1.0, n_steps)
        x_inter_source_to_target = solver.sample(x_init=x_0, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=False)
        # x_inter_source_to_target = solver.sample(x_init=x_0, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        return x_inter_source_to_target

    def backward_flow(self,x_1, solver_type='midpoint', n_steps=10):

        self.model.eval()
        wrapped_model = BatchedVelocityWrapper(self.model)
        solver = ODESolver(velocity_model=wrapped_model)

        time_steps = torch.linspace(1.0, 0.0, n_steps)
        # x_inter_target_to_source = solver.sample(x_init=x_1, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=False)
        x_inter_target_to_source = solver.sample(x_init=x_1, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        return x_inter_target_to_source

    @torch.no_grad()
    def compute_anomaly_scores(self, X_test, X_inlier,
                            type='mahalanobis',
                            n_steps=20,
                            solver_type="midpoint"):

        ##################################
        ########### TEST DATA ############
        ##################################

        if self.noise_is_target:
            # forward ODE : test -> noise
            out = self.forward_flow(X_test.to(self.device), solver_type=solver_type, n_steps=n_steps)
        else:
            # backward ODE : noise -> test
            out = self.backward_flow(X_test.to(self.device), solver_type=solver_type, n_steps=n_steps)

        x_1_test = out[-1].cpu().numpy()

        ##########################################
        ########### INLIERS (TRAIN) ##############
        ##########################################

        if type == 'mahalanobis':

            if self.noise_is_target:
                # forward ODE : inlier -> noise
                out_in = self.forward_flow(X_inlier.to(self.device), solver_type=solver_type, n_steps=n_steps)
            else:
                # backward ODE : noise -> inlier
                out_in = self.backward_flow(X_inlier.to(self.device), solver_type=solver_type, n_steps=n_steps)

            x_1_inliers = out_in[-1].cpu().numpy()

            mean_inlier = np.mean(x_1_inliers, axis=0)
            cov_inlier = np.cov(x_1_inliers.T)
            cov_inlier += 1e-6 * np.eye(cov_inlier.shape[0])

            diff = x_1_test - mean_inlier
            inv_cov = np.linalg.inv(cov_inlier)

            scores = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        ##################################
        ########### NORMS ################
        ##################################
        elif type == 'norm':
            scores = np.sum(x_1_test ** 2, axis=1)

        elif type == 'norm-centroid':
            # scores = np.sum((x_1_test[:, 0, :] - self.centroid.repeat(x_1_test.shape[0],1).cpu().numpy()) ** 2, axis=1)
            scores = np.sum((x_1_test - self.centroid.repeat(x_1_test.shape[0],1).cpu().numpy()) ** 2, axis=1)

        ##################################
        ######## RECONSTRUCTION ##########
        ##################################
        elif type == 'recons':

            if self.noise_is_target:
                out_back = self.backward_flow(torch.tensor(x_1_test, device=self.device), solver_type=solver_type, n_steps=n_steps)
            else:
                out_back = self.forward_flow(torch.tensor(x_1_test, device=self.device), solver_type=solver_type, n_steps=n_steps)

            x_0_back = out_back[-1]
            scores = (
                torch.norm(x_0_back - X_test, dim=1) ** 2
            ).cpu().numpy()

        return scores

    def test(self, X_test, mask, y_test, type='topk', k_rate=None, attentions=None):

        x_final, _, _ = self.euler_integrate(X_test.to(self.device), mask, 15, False)
        # x_final = self.forward_flow(X_test, solver_type='midpoint', n_steps=15)
        s_norm = torch.norm(x_final - self.centroid, dim=-1)

        if type == 'mean':
            score = s_norm.mean(dim=-1)

        if type == 'mediane':
            score = torch.median(s_norm, dim=1).values

        if type == 'sum':
            score = s_norm.sum(dim=1)

        if type == 'topk':
            k = int(x_final.shape[1] * k_rate)
            topk_vals, _ = torch.topk(s_norm, k=min(k, s_norm.shape[-1]), dim=-1)
            topk_vals = topk_vals.clamp(min=0.0)
            score = topk_vals.mean(dim=-1)
        
        if type == 'max':
            score = s_norm.max(dim=-1).values

        if type == 'attention_weighted':
            attn = attentions.mean(dim=1)
            weights = (attn / (attn.sum(dim=-1, keepdim=True) + 1e-8))
            score = (s_norm * weights).mean(axis=-1)

        if type == 'weights':
            attn = attentions.mean(dim=1)
            weights = (attn / (attn.sum(dim=-1, keepdim=True) + 1e-8))
            score = weights.mean(dim=-1)


        return ev.evaluation(y_test, score.cpu().numpy(), True)

def compactness_pairwise(X):
    ll = []
    for x in X:
        dist = torch.cdist(x, x)
        ll.append(dist.mean().item())
        return np.array(ll)