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

        # because the source and target distirbution are inlier and inlier_masked
        self.is_masking_task = type(self.config['source']) != str and type(self.config['target']) != str

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
        
    def sample(self, x_0, type):

        if type == 'gaussian':
            return torch.randn_like(x_0)
        
        if type == 'gaussian-neigh':
            # N(centroid, 0.01 I)
            # centroid = x_0.mean(dim=0, keepdim=True) 

            std = ((self.var * self.config['coef_var']) ** 0.5)
            return self.centroid + std * torch.randn_like(x_0)
        
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


    def compute_flow_loss(self, x_0, indices, flow_type='linear', sigma=0.1, lambda_reg_angle=2.):
            
        batch_size = x_0.shape[0]
        if self.is_masking_task:
            x_1 = self.target[indices]

        if self.target == 'gaussian' or self.source == 'gaussian':
            x_1 = self.sample(x_0, 'gaussian')

        if self.target == 'gaussian-neigh' or self.source == 'gaussian-neigh':
            x_1 = self.sample(x_0, 'gaussian-neigh')

        if self.target == 'centroid' or self.source == 'centroid':
            x_1 = self.sample(x_0, 'centroid')

        if self.target == 'sphere' or self.source == 'sphere':
            x_1 = self.sample(x_0, 'sphere-noised')

        if self.target == 'sphere-noised' or self.source == 'sphere-noised':
            x_1 = self.sample(x_0, 'sphere-noised')

        t = torch.rand(batch_size, device=self.device)
        # t = torch.arange(0, 1, (1/batch_size), device=self.device)

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
        loss = F.mse_loss(v_pred, v_target)

        if lambda_reg_angle is not None:
            loss_regul_angle = (1 - F.cosine_similarity(v_pred, v_target, dim=-1)).pow(2).mean()
            return loss+lambda_reg_angle*loss_regul_angle, loss_regul_angle, v_pred, v_target
        
        else:
            return loss, v_pred, v_target
    

    # def train_epoch(self, dataloader, optimizer):

    #     self.model.train()
    #     total_loss = 0
    #     total_loss_regul = 0
    #     self.optimizer = optimizer
        
    #     for x_0, indices in dataloader:
            
    #         x_0 = x_0.to(self.device)
            
    #         loss, *anythingelse = self.compute_flow_loss( 
    #             x_0, 
    #             indices,
    #             flow_type=self.config['flow_type'],
    #             sigma=self.config['sigma'],
    #             lambda_reg_angle = self.config['lambda_reg_angle']
    #         )

    #         if self.config['lambda_reg_angle'] is not None:
    #             loss_regul = anythingelse[0]
    #             total_loss_regul += loss_regul.item()

    #         self.optimizer.zero_grad()
    #         loss.backward()
    
    #         torch.nn.utils.clip_grad_norm_(
    #             self.model.parameters(), 
    #             self.config['grad_clip']
    #         )
            
    #         self.optimizer.step()
            
    #         total_loss += loss.item()

    #     avg_loss = total_loss / len(dataloader)
    #     avg_loss_regul = total_loss_regul / len(dataloader)
        
    #     return avg_loss, avg_loss_regul


    def train_epoch(self, dataloader, optimizer, epoch):

        self.model.train()
        total_loss = 0
        total_loss_regul = 0
        self.optimizer = optimizer

        for x_0, indices in dataloader:

            x_0 = x_0.to(self.device)

            loss, *anythingelse = self.compute_flow_loss(
                x_0,
                indices,
                flow_type=self.config['flow_type'],
                sigma=self.config['sigma'],
                lambda_reg_angle=self.config['lambda_reg_angle']
            )

            if self.config['lambda_reg_angle'] is not None:
                loss_regul = anythingelse[0]
                total_loss_regul += loss_regul.item()

            ###########################################
            ############### REGUL #####################
            if self.config['lambda_reg_kl'] is not None:
                target_constructed_epoch = self.forward_flow(
                    x_0, 'midpoint', 10
                )[-1]  

                mu_constructed = target_constructed_epoch.mean(dim=0)
                var_constructed = target_constructed_epoch.var(dim=0, unbiased=False)

                mu_target = self.centroid
                var_target = self.var / self.config['coef_var']

                # KL diagonale analytique
                kl_loss = 0.5 * torch.sum(
                    torch.log(var_target / var_constructed)
                    + (var_constructed + (mu_constructed - mu_target)**2) / var_target
                    - 1
                )
                loss = loss + self.config['lambda_reg_kl']* kl_loss
            ###########################################

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )

            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_loss_regul = total_loss_regul / len(dataloader)

        return avg_loss, avg_loss_regul

    
    def train(self, X_inlier, verbose=True):

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            foreach=False
        )

        X_inlier_dl = DataLoader(TensorDataset(X_inlier, torch.arange(len(X_inlier))), batch_size=self.config['batch_size'], shuffle=True)

        liste_loss = []
        liste_loss_regul = []

        for epoch in range(self.config['epochs']):
            
            lr = self.get_lr_schedule(epoch, self.config['warmup_epochs'], self.config['epochs'], self.config['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            train_loss, *anythingelse = self.train_epoch(
                X_inlier_dl, 
                optimizer, 
                epoch
            )

            if self.config['lambda_reg_angle'] is not None:
                liste_loss_regul.append(anythingelse[0])
            
            liste_loss.append(train_loss)

            if epoch % (self.config['epochs'] // 3) == 0 and verbose:
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                print(f"Train Loss: {train_loss:.4f}, LR: {lr:.6f}")

        if self.rectified is None:
            return liste_loss, liste_loss_regul
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

            return liste_loss, liste_loss_regul

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


class BatchedVelocityWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        # t est scalaire → on l'étend au batch
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t)

def kl_gaussian(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    KL( N1 || N2 )
    mu1, mu2 : (d,)
    cov1, cov2 : (d, d)
    """

    d = mu1.shape[0]

    # Stabilisation
    cov1 = cov1 + eps * torch.eye(d, device=mu1.device)
    cov2 = cov2 + eps * torch.eye(d, device=mu1.device)

    inv_cov2 = torch.linalg.inv(cov2)

    logdet_cov1 = torch.logdet(cov1)
    logdet_cov2 = torch.logdet(cov2)

    trace_term = torch.trace(inv_cov2 @ cov1)

    mean_diff = (mu2 - mu1).unsqueeze(0)  # (1, d)
    mahalanobis = mean_diff @ inv_cov2 @ mean_diff.T

    kl = 0.5 * (
        logdet_cov2
        - logdet_cov1
        - d
        + trace_term
        + mahalanobis.squeeze()
    )

    return kl




    # @torch.no_grad()
    # def compute_anomaly_scores(self, X_test, X_inlier, type='mahalanobis' , n_steps=100):

    #     self.model.eval()
    #     ##################################
    #     ############ Testing Data ########
    #     ##################################

    #     if self.noise_is_target:
    #         # forward ODE : test -> noise
    #         x_t = X_test.to(self.device)
    #         t_schedule = torch.linspace(0.0, 1.0, n_steps, device=self.device)
    #         sign = +1.0
    #     else:
    #         # backward ODE : noise -> test
    #         x_t = self.sample(X_test, self.source)
    #         t_schedule = torch.linspace(1.0, 0.0, n_steps, device=self.device)
    #         sign = -1.0

    #     delta_t = 1.0 / n_steps
    #     velo_s = []

    #     for t_scalar in t_schedule:
    #         t = torch.full((x_t.shape[0],), t_scalar, device=self.device)
    #         v = self.model(x_t, t)
    #         velo_s.append(v)
    #         x_t = x_t + sign * v * delta_t

    #     x_1_test = x_t.cpu().numpy()

    #     if type == 'mahalanobis':

    #         ##########################################
    #         ############ Training Inlier Data ########
    #         ##########################################
            
            
    #         if self.noise_is_target:
    #             # forward ODE : inliers -> noise
    #             x_t = X_inlier.to(self.device)
    #             t_schedule = torch.linspace(0.0, 1.0, n_steps, device=self.device)
    #             sign = +1.0
    #         else:
    #             # backward ODE : noise -> inliers
    #             x_t = self.sample(X_inlier, self.source)
    #             t_schedule = torch.linspace(1.0, 0.0, n_steps, device=self.device)
    #             sign = -1.0

    #         delta_t = 1.0 / n_steps

    #         for t_scalar in t_schedule:
    #             t = torch.full((x_t.shape[0],), t_scalar, device=self.device)
    #             v = self.model(x_t, t)
    #             x_t = x_t + sign * v * delta_t

    #         x_1_inliers = x_t.cpu().numpy()

            
    #         mean_inlier = np.mean(x_1_inliers, axis=0)
    #         cov_inlier = np.cov(x_1_inliers.T)
            
    #         cov_inlier += 1e-6 * np.eye(cov_inlier.shape[0])

    #         # mahalanobis score ---> sqrt((x - μ)^T Σ^(-1) (x - μ))
    #         diff = x_1_test - mean_inlier
    #         inv_cov = np.linalg.inv(cov_inlier)
            
    #         scores = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    #     if type == 'norm':
    #         scores = np.sum((x_1_test ** 2), axis=1)

    #     if type == 'norm-centroid':
    #         centroid = X_inlier.mean(dim=0).cpu().numpy()
    #         scores = np.sum(((x_1_test - centroid) ** 2), axis=1)

    #     if type == 'recons':

    #         x_t = Tensor(x_1_test).to(self.device)

    #         if self.noise_is_target:
    #             t_schedule = torch.linspace(1.0, 0.0, n_steps, device=self.device)
    #             sign = -1.0
    #         else:
    #             t_schedule = torch.linspace(0.0, 1.0, n_steps, device=self.device)
    #             sign = +1.0

    #         delta_t = 1.0 / n_steps

    #         for t_scalar in t_schedule:
    #             t = torch.full((x_t.shape[0],), t_scalar, device=self.device)
    #             v = self.model(x_t, t)
    #             x_t = x_t + sign * v * delta_t

    #         x_0_test_back = x_t

    #         scores = ((torch.norm(x_0_test_back - X_test, dim=1)** 2)).cpu().detach()

    #     return scores, velo_s