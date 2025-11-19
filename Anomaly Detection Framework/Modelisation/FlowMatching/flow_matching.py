
import numpy as np
from sklearn.datasets import make_moons,make_circles
import torch
from torch import nn, Tensor
import math
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from Modelisation.FlowMatching.utils import anomaly_score

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, device, max_period=10000):
        super().__init__()
        self.dim = dim
        self.device = device
        half_dim = dim // 2
        self.freq = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(self.device)
        # self.register_buffer("freq", freq)

    def forward(self, t):
        # t: [B, 1]
        t = t.view(-1, 1)
        args = t * self.freq  # [B, half_dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


class FlowMatching(nn.Module):
    def __init__(self, source, target, input_dim=64, latent_dim=256, sinusoidal=False, device='cuda'):
        super().__init__()
        
        self.target = target
        self.source = source
        self.device = device
        self.sinusoidal = sinusoidal
        
        if self.source == 'MoG' : 
            n_components = 2
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(self.target)
            self.gmm = gmm
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # adding the time dimension
        first_dim = input_dim + 1
        
        if sinusoidal : 
            self.time_embedding = SinusoidalTimeEmbedding(self.input_dim, self.device).to(self.device)
            # because of concatenation of time embedding with x
            first_dim = input_dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(first_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, input_dim)
        )
        
    def forward(self, x, t):
        # t = t.expand(x.shape[0], 1)
        if self.sinusoidal:
            t = self.time_embedding(t).to(self.device)
            
        xt = torch.cat([x, t], dim=1)
        # xt = torch.cat([x, t], dim=1)
        return self.net(xt)
    
    def sampling_source(self, n_samples):
        
        if self.source == 'gaussian':
            return torch.randn(n_samples, self.input_dim).to(self.device)
        
        if self.source == 'circle': return Tensor(make_circles(n_samples=n_samples, noise=0.1, factor=0.5)[0]).to(self.device)
    
        if self.source == 'MoG': return Tensor(self.gmm.sample(n_samples)[0]).to(self.device)
    
        if self.source == 'poisson': return Tensor(np.random.poisson(5, (n_samples, self.input_dim))).to(self.device)
    
        if self.source == 'uniform': return Tensor(np.random.uniform(low=self.target.min(), high=self.target.max(), size=(n_samples, self.input_dim))).to(self.device)
    
        if self.source == 'sphere':
            
            z = torch.randn(n_samples, self.input_dim)
            return Tensor(z / z.norm(dim=1, keepdim=True)).to(self.device)
        
        if self.source == 'sphere-noised':
            z = torch.randn(n_samples, self.input_dim)
            z = z / z.norm(dim=1, keepdim=True)
            noise = torch.randn_like(z) * 0.3
            return Tensor(z + noise).to(self.device)

    def interpolation(self, n_samples):
        
        # source sampling
        x0 = self.sampling_source(n_samples)
        
        # target sampling ( get n_samples examples from the all target dataset )
        # replace --> avec ou sans remise
        idx = np.random.choice(self.target.shape[0], n_samples, replace=True)
        x1 = self.target[idx].to(self.device)
        
        # sampling the time between 0 ad 1
        t = torch.rand(n_samples, 1).to(self.device)

        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0
        
        return xt, t, ut
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2,
        t_start + (t_end - t_start) / 2)
    


class FlowMatchingTrainer():
    def __init__(self, flow_model, optimizer, loss_fn, n_steps, batch_size, verbose=True):

        self.flow_model = flow_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose

    def train(self):

        if self.verbose: recons_err = []

        for s in range(self.n_steps):

            xt, t, ut = self.flow_model.interpolation(self.batch_size)

            vt =  self.flow_model(xt, t)

            if self.verbose:
                x_recons = xt - vt * t
                recons_err.append(torch.norm(xt - x_recons).item())

            self.optimizer.zero_grad()
            loss = self.loss_fn(vt, ut)

            if self.verbose and s % (self.n_steps//10) == 0: 
                print(f" step {s} -> loss : {loss.item():.5f}, recon_err : {np.mean(recons_err):.5f}")
                recons_err = []

            loss.backward()
            self.optimizer.step()

    def test(self, flow_model, X_test, y_test):
        scores = anomaly_score(X_test, flow_model, n_steps=100)[2].cpu().detach()

        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        fpr, tpr, thresholds = roc_curve(y_test, scores)
        idx = np.where(tpr >= 0.95)[0][0]
        fpr95 = fpr[idx]

        print(f"AUC: {auc:.4f} | FPR@95: {fpr95:.4f} | AP: {ap:.4f}")  

        return auc, fpr95, ap 

    