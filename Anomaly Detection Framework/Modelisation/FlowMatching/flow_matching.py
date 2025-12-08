
import numpy as np
from sklearn.datasets import make_moons,make_circles
import torch
from torch import nn, Tensor
import math
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
# from Modelisation.FlowMatching.utils import anomaly_score
from flow_matching.solver import ODESolver

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
    def __init__(self, source, target, input_dim=64, latent_dim=256, sinusoidal=False, dropout=0.3, batchnorm=False, device='cuda', seed=42):
        super().__init__()
        
        self.seed = seed
        self.target = target
        self.target_centroid = self.target.mean(dim=0)
        self.target_cov_mat = torch.cov(self.target.T)
        self.source = source
        self.device = device
        self.sinusoidal = sinusoidal
        
        if self.source == 'MoG' : 
            n_components = 2
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(self.target)
            self.gmm = gmm
            
        if self.source == 'target-noised':
            noise = torch.randn(self.target.shape[0], self.target.shape[1])
            self.target_noised = self.target + noise
            
        if self.source == 'point-mean':
            self.point_mean = self.target.mean(dim=0).to(self.device)
            
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # adding the time dimension
        first_dim = input_dim + 1
        
        if sinusoidal : 
            self.time_embedding = SinusoidalTimeEmbedding(self.input_dim, self.device).to(self.device)
            # because of concatenation of time embedding with x
            first_dim = input_dim * 2
        
        layers = [nn.Linear(first_dim, latent_dim)]
        
        if batchnorm : 
            layers.append(nn.BatchNorm1d(latent_dim))
            
        layers.append(nn.Tanh())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(latent_dim, latent_dim))
        
        if batchnorm : 
            layers.append(nn.BatchNorm1d(latent_dim))
        layers.append(nn.Tanh())
                
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
            
        layers.append(nn.Linear(latent_dim, input_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t = t.expand(x.shape[0], 1)
        if self.sinusoidal:
            t = self.time_embedding(t).to(self.device)
            
        xt = torch.cat([x, t], dim=1)

        return self.net(xt)
    
    def sampling_source(self, n_samples):

        # torch.manual_seed(self.seed) 
        # torch.cuda.manual_seed_all(self.seed)
        
        if self.source == 'gaussian':
            return torch.randn(n_samples, self.input_dim).to(self.device)
#             eps = 1e-2
#             cov_reg = self.target_cov_mat + eps * torch.eye(self.target_cov_mat.size(0))

#             distri = torch.distributions.MultivariateNormal(self.target_centroid, covariance_matrix=cov_reg)
#             return distri.sample((n_samples,)).to(self.device)
        
        if self.source == 'circle': return Tensor(make_circles(n_samples=n_samples, noise=0.1, factor=0.5)[0]).to(self.device)
    
        if self.source == 'MoG': return Tensor(self.gmm.sample(n_samples)[0]).to(self.device)
    
        if self.source == 'poisson': return Tensor(np.random.poisson(5, (n_samples, self.input_dim))).to(self.device)
    
        if self.source == 'uniform': return Tensor(np.random.uniform(low=self.target.min(), high=self.target.max(), size=(n_samples, self.input_dim))).to(self.device)
    
        if self.source == 'sphere':
            
            z = torch.randn(n_samples, self.input_dim)
            return Tensor(z / z.norm(dim=1, keepdim=True)).to(self.device)
        
        if self.source == 'sphere-noised':
            z = torch.randn(n_samples, self.input_dim)
            
            
#             eps = 1e-2
#             cov_reg = self.target_cov_mat + eps * torch.eye(self.target_cov_mat.size(0))

#             distri = torch.distributions.MultivariateNormal(self.target_centroid, covariance_matrix=cov_reg)
#             z = distri.sample((n_samples,)).to(self.device)
            
            z = z / z.norm(dim=1, keepdim=True)
            noise = torch.randn_like(z) * 0.25
            return Tensor(z + noise).to(self.device)
        
        if self.source == 'point-mean':
            return self.point_mean.repeat(n_samples).reshape(-1,self.input_dim)
            
        
    def interpolation(self, n_samples, x1=None):

        # source sampling
        x0 = self.sampling_source(n_samples)
        
        # target sampling ( get n_samples examples from the all target dataset )
        # replace --> avec ou sans remise
        if x1 is None : idx = np.random.choice(self.target.shape[0], n_samples, replace=True)
        
        if self.source == 'target-noised':
            x0 = self.target_noised[idx].to(self.device)
        
        if x1 is None : x1 = self.target[idx].to(self.device)

        # sampling the time between 0 ad 1
        t = torch.rand(n_samples, 1).to(self.device)
        # t = torch.rand(n_samples).to(self.device)
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0
        
        return xt, t, ut



class FlowMatchingTrainer():
    def __init__(self, flow_model,  verbose=True):

        self.flow_model = flow_model
        self.verbose = verbose

    def train(self, dataloader, lr, weight_decay, loss_fn, n_epochs, optimizer_type='adam'):
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=weight_decay)

        for s in range(n_epochs):

            for data, in dataloader:

                xt, t, ut = self.flow_model.interpolation(data.shape[0], data)
                vt =  self.flow_model(xt, t)

                optimizer.zero_grad()

                loss = loss_fn(vt, ut)
                loss.backward()

                optimizer.step()
            
            if self.verbose and s % (n_epochs // 5) == 0:
                print(f" step {s} -> loss : {loss.item():.5f}")
                
        return self.flow_model

    def forward_flow(self, x_0, solver_type='midpoint', n_steps=10):
            
        solver = ODESolver(velocity_model=self.flow_model)

        time_steps = torch.linspace(0.0, 1.0, n_steps)
        x_inter_source_to_target = solver.sample(x_init=x_0, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        return x_inter_source_to_target

    def backward_flow(self,x_1, solver_type='midpoint', n_steps=10):
        solver = ODESolver(velocity_model=self.flow_model)

        time_steps = torch.linspace(1.0, 0.0, n_steps)
        x_inter_target_to_source = solver.sample(x_init=x_1, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        return x_inter_target_to_source
    
    def forward_backward_flow(self, x_1, solver_type='midpoint', n_steps=10):
        solver = ODESolver(velocity_model=self.flow_model)

        time_steps = torch.linspace(1.0, 0.0, n_steps)
        x_inter_target_to_source = solver.sample(x_init=x_1, method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        time_steps = torch.linspace(0.0, 1.0, n_steps)
        x_inter_source_to_target_recons = solver.sample(x_init=x_inter_target_to_source[-1], method=solver_type, step_size=1.0 / n_steps, time_grid=time_steps, return_intermediates=True)

        return x_inter_source_to_target_recons


    def test(self, X_test, y_test, score_type='norm', solver_type='midpoint', n_steps=10):

        if score_type == 'norm':
            x_source_after_backward = self.backward_flow(X_test, solver_type, n_steps)[-1].cpu().detach()
            # scores = ((x_source_after_backward - self.flow_model.target_centroid) ** 2).sum(dim=1)
            scores = (x_source_after_backward ** 2).sum(dim=1)

        if score_type == 'recons':
            x_target_after_forward_backward = self.forward_backward_flow(X_test, solver_type, n_steps)[-1]
            scores = ((torch.norm(x_target_after_forward_backward - X_test, dim=1)** 2)).cpu().detach()
 
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        fpr, tpr, thresholds = roc_curve(y_test, scores)
        idx = np.where(tpr >= 0.95)[0][0]
        fpr95 = fpr[idx]

        print(f"FM --> AUC: {auc:.4f} | FPR@95: {fpr95:.4f} | AP: {ap:.4f}")  

        return auc, fpr95, ap 
