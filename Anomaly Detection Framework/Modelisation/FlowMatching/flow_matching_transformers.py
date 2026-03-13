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
from scipy.stats import chi2

class BatchedVelocityWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        # t est scalaire → on l'étend au batch
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: Tensor (batch,) ou (batch, 1)
        if x.dim() == 2:
            x = x.squeeze(-1)

        device = x.device
        half_dim = self.hidden_dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -emb_scale
        )

        x = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    

class FlowDiT(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=64, depth=2, n_heads=2):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Blocs DiT avec adaLN
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads) for _ in range(depth)
        ])
        
        # Tête de prédiction du champ de vecteurs
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, t):
        # x: [batch, latent_dim], t: [batch]
        h = self.input_proj(x)
        t_emb = self.time_mlp(t)
        
        for block in self.blocks:
            # adaLN conditioning
            h = block(h, t_emb)  
        
        v = self.output_proj(h)
        return v
    

class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # adaLN: prédit scale et shift à partir du timestep
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)  
        )
    
    def forward(self, x, t_emb):
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(t_emb).chunk(6, dim=-1)
                
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        # ajoute une dimension sequence (self-attention sur un seul token)
        x_norm = x_norm.unsqueeze(1)  
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  
        attn_out = attn_out.squeeze(1)  
        x = x + gate_msa * attn_out
        
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FlowMatchingTransformers(nn.Module):

    def __init__(self, model, source, target, config, noise_is_target, rectified):
        super().__init__()
        self.model = model
        self.source = source
        self.target = target
        self.noise_is_target = noise_is_target
        if self.noise_is_target:
            self.device = self.source.device
            self.centroid = Tensor(self.source.mean(dim=0)).to(self.device)
            self.var = Tensor(self.source.var(dim=0)).mean().to(self.device)
            # self.centroid = torch.ones((self.source.shape[1]), device=self.device)*2
        else:
            self.device = self.target.device
            self.centroid = Tensor(self.target.mean(dim=0)).to(self.device)
            self.var = Tensor(self.target.var(dim=0)).mean().to(self.device)
            # self.centroid = torch.ones((self.target.shape[1]), device=self.device)*2
        self.rectified = rectified
        self.config = config

        self.log_r = nn.Parameter(torch.tensor(0.0).to(self.device))

        # dist_train = torch.norm(self.source - self.centroid, dim=1)
        # r_init = torch.quantile(dist_train, 0.95)
        # self.log_r = nn.Parameter(torch.log(r_init))
        # print(self.log_r)


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
            return lr * (epoch + 1) / warmup_epochs
        # cosinus decrease
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        

    def get_lambda_kl(self, epoch, lambda_max, kl_warmup_epochs):
        return lambda_max * min(1.0, epoch / kl_warmup_epochs)
          
    def sample_like(self, x_0, type, tail=None):

        if type == 'gaussian':
            return torch.randn_like(x_0)
                
        if type == 'gaussian-neigh':

            sigma = torch.sqrt(self.var * self.config['coef_var'])            
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


    def compute_flow_loss(self, x_0, indices, flow_type='linear', sigma=0.1, warmup_activated=False):
            

        ##########################################
        ############ NEGATIVE SAMPLING ###########
        ##########################################

        if not warmup_activated:
            
            # garde fou
            nb_samples_neg = int((self.config['rate_neg_batch'] * x_0.shape[0]) / 3)
            if nb_samples_neg < 1:
                nb_samples_neg = 5

            x_0_negative = []
            sigma_levels = torch.tensor(self.config['sig_levels_neg']).to(self.device) * torch.sqrt(self.var)
            for i,sig in enumerate(sigma_levels):

                eps = sig * torch.randn((i+1)*nb_samples_neg, x_0.shape[1]).to(self.device)
                x_0_negative.extend(self.centroid + eps)

            x_0_negative = torch.stack(x_0_negative).to(self.device)

            # sigma_levels = [
            #     # 0.9 * torch.sqrt(self.var),
            #     # 1.3 * torch.sqrt(self.var)

            #     0.5 * torch.sqrt(self.var),
            #     0.7 * torch.sqrt(self.var)
            #     # 0.7 * torch.sqrt(self.var),
            #     # 0.9 * torch.sqrt(self.var)

            #     # 0.3 * torch.sqrt(self.var),
            #     # 0.5 * torch.sqrt(self.var)
            # ]

            # alpha = 1.2
            # direction = x_0 - self.centroid       
            # all_x_0_negative = self.centroid + alpha * direction
            # x_0_negative = all_x_0_negative[torch.randint(0,x_0.shape[0],(nb_samples_neg,))]
            

        ##########################################
        ##########################################

        if self.target == 'gaussian' or self.source == 'gaussian':
            x_1 = self.sample_like(x_0, 'gaussian')

        if self.target == 'gaussian-neigh' or self.source == 'gaussian-neigh':
            x_1 = self.sample_like(x_0, 'gaussian-neigh')

        if self.target == 'centroid' or self.source == 'centroid':
            x_1 = self.sample_like(x_0, 'centroid')

        if self.target == 'sphere' or self.source == 'sphere':
            x_1 = self.sample_like(x_0, 'sphere-noised')

        if self.target == 'sphere-noised' or self.source == 'sphere-noised':
            x_1 = self.sample_like(x_0, 'sphere-noised')

        batch_size = x_1.shape[0]
        t = torch.rand(batch_size, device=self.device)

        # j'inverse juste la source et la target
        if not self.noise_is_target:
            x_0, x_1 = x_1, x_0

        if flow_type == 'linear':
            
            t_expanded = t.view(-1, 1)
            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
            v_target = x_1 - x_0
            
        elif flow_type == 'cfm':
            t_expanded = t.view(-1, 1)
            mu_t = t_expanded * x_1 + (1 - t_expanded) * x_0
            
            eps = torch.randn_like(x_0)
            x_t = mu_t + sigma * eps
            
            v_target = x_1 - x_0
        
        else:
            raise ValueError(f"Unknown flow_type: {flow_type}")
        
        v_pred = self.model(x_t, t)

        # <<<<<<<<<<<<<<<<<<<<< LOSS FM >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        loss_fm = F.mse_loss(v_pred, v_target)


        if not warmup_activated:

            # <<<<<<<<<<<<<<<<<<<<< LOSS SVDD >>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            lambda_svdd = 1e-3
            
            t_zeros = torch.zeros(x_0.shape[0], device=self.device)  
            v_svdd = self.model(x_0, t_zeros)                        
            x_svdd = x_0 + v_svdd                                    

            dist_sq = torch.sum((x_svdd - self.centroid)**2, dim=1)    
            r_sq = self.r_in ** 2                                         
            loss_svdd = r_sq + F.relu(dist_sq - r_sq).mean()    


            # <<<<<<<<<<<<<<<<<<<<< LOSS PUSH >>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            t_neg = torch.zeros(x_0_negative.shape[0]).to(self.device)
            v_neg = self.model(x_0_negative, t_neg)
            x_neg = x_0_negative + v_neg 
            dist_out = torch.norm(x_neg - self.centroid, dim=1)         
            loss_push = F.relu(self.r_out - dist_out).mean()
            lambda_push = 1e-2


            # <<<<<<<<<<<<<<<<<<<<< REGUL MARGIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    
            lambda_margin = 1e-2
            
            loss_total = loss_fm + lambda_svdd * loss_svdd + lambda_push * loss_push + lambda_margin * self.margin
            return loss_total, loss_fm.item(), loss_svdd.item(), loss_push.item(), self.margin.item(), self.r_in.item()

        else:

            loss_total = loss_fm
            return loss_total, 0, 0, 0, 0, 0

        # print(f"Loss FM : {loss_fm}")
        # print(f"Loss SVDD : {loss_svdd}")
        # print(f"Loss Push : {loss_push}")
        # print(f"Loss Tot : {loss_total}")
        # print("----------------------------------------")



    def train_epoch(self, dataloader, optimizer, warmup_activated):

        self.model.train()
        self.optimizer = optimizer

        total_loss_liste = []
        loss_fm_liste = []
        loss_svdd_liste = []
        loss_push_liste = []
        margin_value_liste = []
        r_in_value_liste = []

        # for x_0, y, indices in dataloader:
        for x_0, indices in dataloader:

            x_0 = x_0.to(self.device)
            loss_total, *anythingelse = self.compute_flow_loss(
            # loss_total, loss_fm, loss_svdd, loss_push, margin_value, r_in_value = self.compute_flow_loss(
                x_0,
                indices,
                flow_type=self.config['flow_type'],
                sigma=self.config['sigma'],
                warmup_activated=warmup_activated
            )

            self.optimizer.zero_grad()
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )

            self.optimizer.step()

            total_loss_liste.append(loss_total.item())
            loss_fm_liste.append(anythingelse[0])
            loss_svdd_liste.append(anythingelse[1])
            loss_push_liste.append(anythingelse[2])
            margin_value_liste.append(anythingelse[3])
            r_in_value_liste.append(anythingelse[4])

        return np.mean(total_loss_liste), np.mean(loss_fm_liste), np.mean(loss_svdd_liste),\
              np.mean(loss_push_liste),  np.mean(margin_value_liste), np.mean(r_in_value_liste)

    
    def train(self, verbose=True):

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
        X_inlier_dl = DataLoader(TensorDataset(X_inlier, torch.arange(len(X_inlier))), batch_size=self.config['batch_size'], shuffle=True)


        total_loss_liste = []
        loss_fm_liste = []
        loss_svdd_liste = []
        loss_push_liste = []
        margin_value_liste = []
        r_in_value_liste = []


        for epoch in range(self.config['epochs']):

            warmup_activated = epoch <= self.config['warmup_epochs']

            if epoch == self.config['warmup_epochs']:
                self.model.eval()
                with torch.no_grad():
                    t_zeros = torch.zeros(X_inlier.shape[0]).to(self.device)
                    v = self.model(X_inlier, t_zeros)
                    phi_1 = X_inlier + v

                dist_train = torch.norm(phi_1 - self.centroid, dim=1)
                r_init = torch.quantile(dist_train, 0.90)
                self.log_r.data.fill_(torch.log(r_init).item())
                print(f"FM Warmup is finished.... initialization r_in : {self.r_in}")
            
            lr = self.get_lr_schedule(epoch, self.config['warmup_epochs'], self.config['epochs'], self.config['lr'])
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
            loss_push_liste.append(anythingelse[2])
            margin_value_liste.append(anythingelse[3])
            r_in_value_liste.append(anythingelse[4])
        

            if epoch % (self.config['epochs'] // 3) == 0 and verbose:
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                print(f"Train Loss: {loss_total:.4f}, LR: {lr:.6f}")

        if self.rectified is None:
            return total_loss_liste, loss_fm_liste, loss_svdd_liste, \
                    loss_push_liste, margin_value_liste, r_in_value_liste

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
        x_inter_source_to_target = solver.sample(x_init=x_0, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        return x_inter_source_to_target

    def backward_flow(self,x_1, solver_type='midpoint', n_steps=10):

        self.model.eval()
        wrapped_model = BatchedVelocityWrapper(self.model)
        solver = ODESolver(velocity_model=wrapped_model)

        time_steps = torch.linspace(1.0, 0.0, n_steps)
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

    def test(self, X_test, y_test, X_inlier, type='mahalanobis'):

        scores = self.compute_anomaly_scores(X_test, X_inlier, type)
        return ev.evaluation(y_test, scores, verbose=False)


def energy(phi_1, centroid, sigma):
    dist_sq = torch.sum((phi_1 - centroid)**2, dim=1)  # (B,)
    return torch.exp(-dist_sq / (2 * sigma**2))   
    # return dist_sq.mean() / (2 * sigma**2 )   