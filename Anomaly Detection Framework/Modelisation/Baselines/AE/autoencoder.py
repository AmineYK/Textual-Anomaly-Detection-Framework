from Modelisation.Baselines.baseline import BaselineModel
import Modelisation.evaluation as ev
from pyod.models.auto_encoder import AutoEncoder

class AE(BaselineModel):

    def __init__(self, args):
        self.model = AutoEncoder(**args)

    def train(self, X_train):
        
        self.model.fit(X_train.cpu())

        return self.model

    def test(self, X_test, y_test):        

        scores = self.model.decision_function(X_test.cpu())

        auc_ae, fpr95_ae, ap_ae = ev.evaluation(y_test, scores, verbose=False)

        return auc_ae, fpr95_ae, ap_ae 







# import Modelisation.evaluation as ev
# from Modelisation.Baselines.baseline import BaselineModel
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class AE(nn.Module):
#     def __init__(self, input_dim, hidden_layer_sizes, intrinsic_size, 
#                  norm_type="MSE", loss_norm_type="MSE",
#                  activation=F.relu, if_rsr=False, enforce_proj=False, all_alt=False,
#                  learning_rate=1e-3, lambda1=0.1, lambda2=0.1,
#                  epoch_size=50, batch_show=25, normalize=False, bn=True, seed=123):
#         super().__init__()
        
#         self.input_dim = int(input_dim)
#         self.hidden_layer_sizes = [int(h) for h in hidden_layer_sizes]
#         self.intrinsic_size = int(intrinsic_size)
#         self.activation = activation
#         self.norm_type = norm_type
#         self.loss_norm_type = loss_norm_type
#         self.learning_rate = learning_rate
#         self.epoch_size = epoch_size
#         self.batch_show = batch_show
#         self.normalize = normalize
#         self.bn = bn
        
#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)
        
#         # --- Encoder ---
#         self.encoder_fc1 = nn.Linear(self.input_dim, self.hidden_layer_sizes[0])
#         self.encoder_fc2 = nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
#         self.encoder_fc3 = nn.Linear(self.hidden_layer_sizes[1], self.hidden_layer_sizes[2])
        
#         if bn:
#             self.bn1 = nn.BatchNorm1d(self.hidden_layer_sizes[0])
#             self.bn2 = nn.BatchNorm1d(self.hidden_layer_sizes[1])
#             self.bn3 = nn.BatchNorm1d(self.hidden_layer_sizes[2])
        
#         # --- NO RSR LAYER HERE ---
        
#         # --- Decoder ---
#         self.decoder_fc3 = nn.Linear(self.hidden_layer_sizes[2], self.hidden_layer_sizes[2])
#         self.decoder_fc2 = nn.Linear(self.hidden_layer_sizes[2], self.hidden_layer_sizes[1])
#         self.decoder_fc1 = nn.Linear(self.hidden_layer_sizes[1], self.hidden_layer_sizes[0])
#         self.decoder_fc0 = nn.Linear(self.hidden_layer_sizes[0], self.input_dim)
        
#         if bn:
#             self.dbn1 = nn.BatchNorm1d(self.hidden_layer_sizes[2])
#             self.dbn2 = nn.BatchNorm1d(self.hidden_layer_sizes[1])
#             self.dbn3 = nn.BatchNorm1d(self.hidden_layer_sizes[0])

#     def encoder(self, x):
#         z = self.activation(self.encoder_fc1(x))
#         if self.bn: z = self.bn1(z)
#         z = self.activation(self.encoder_fc2(z))
#         if self.bn: z = self.bn2(z)
#         z = self.activation(self.encoder_fc3(z))
#         if self.bn: z = self.bn3(z)
#         return z

#     def decoder(self, z):
#         z = self.activation(self.decoder_fc3(z))
#         if self.bn: z = self.dbn1(z)
#         z = self.activation(self.decoder_fc2(z))
#         if self.bn: z = self.dbn2(z)
#         z = self.activation(self.decoder_fc1(z)) 
#         x_hat = self.decoder_fc0(z)
#         return x_hat
    
#     def forward(self, x):
#         y = self.encoder(x)
#         z = y                       # no RSR
#         x_hat = self.decoder(z)
#         return y, z, x_hat

#     # --- Reconstruction loss ONLY ---
#     def reconstruction_error(self, x, x_hat):
#         if self.loss_norm_type.lower() in ['mse','f','frob']:
#             return torch.mean(torch.norm(x - x_hat, dim=1)**2)
#         elif self.loss_norm_type.lower() in ['l1']:
#             return torch.mean(torch.norm(x - x_hat, p=1, dim=1))
#         else:
#             return torch.mean(torch.norm(x - x_hat, dim=1))

#     # --- Training ---
#     def fit(self, X, batch_size=128, x_val=None, device='cuda'):
#         self.to(device)
#         n_samples = X.shape[0]
#         n_batch = (n_samples - 1) // batch_size + 1
        
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
#         for epoch in range(self.epoch_size):
#             idx = np.random.permutation(n_samples)
#             for batch_idx in range(n_batch):
#                 i_start = batch_idx * batch_size
#                 i_end = min((batch_idx + 1) * batch_size, n_samples)
#                 x_batch = X[idx[i_start:i_end]]

#                 optimizer.zero_grad()
#                 _, z, x_hat = self.forward(x_batch)
                
#                 loss = self.reconstruction_error(x_batch, x_hat)
#                 loss.backward()
#                 optimizer.step()
            
#             # Display
#             if self.batch_show is not None and (epoch+1) % self.batch_show == 0:
#                 if x_val is not None:
#                     with torch.no_grad():
#                         x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
#                         _, _, x_hat_val = self.forward(x_val_t)
#                         val_loss = self.reconstruction_error(x_val_t, x_hat_val)
#                         print(f"Epoch {epoch+1}/{self.epoch_size} - Val Loss: {val_loss.item():.4f}")
#                 else:
#                     print(f"Epoch {epoch+1}/{self.epoch_size} - Loss: {loss.item():.4f}")

#     def get_latent(self, X, device='cuda'):
#         self.to(device)
#         with torch.no_grad():
#             _, z, _ = self.forward(X)
#         return z.cpu().numpy()
    
#     def get_output(self, X, device='cuda'):
#         self.to(device)
#         with torch.no_grad():
#             _, _, x_hat = self.forward(X)
#         return x_hat.cpu().numpy()



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class AE_NoLatent(nn.Module):
#     def __init__(self, input_dim, hidden_layer_sizes,
#                  activation=F.relu,
#                  loss_norm_type="MSE",
#                  learning_rate=1e-3,
#                  epoch_size=50,
#                  batch_show=25,
#                  bn=True,
#                  seed=123):

#         super().__init__()

#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)

#         self.input_dim = input_dim
#         self.h = hidden_layer_sizes
#         self.activation = activation
#         self.loss_norm_type = loss_norm_type
#         self.learning_rate = learning_rate
#         self.epoch_size = epoch_size
#         self.batch_show = batch_show
#         self.bn = bn

#         # ----- Encoder -----
#         self.encoder_fc1 = nn.Linear(input_dim, self.h[0])
#         self.encoder_fc2 = nn.Linear(self.h[0], self.h[1])
#         self.encoder_fc3 = nn.Linear(self.h[1], self.h[2])

#         if bn:
#             self.bn1 = nn.BatchNorm1d(self.h[0])
#             self.bn2 = nn.BatchNorm1d(self.h[1])
#             self.bn3 = nn.BatchNorm1d(self.h[2])

#         # ----- Decoder -----
#         self.decoder_fc3 = nn.Linear(self.h[2], self.h[1])
#         self.decoder_fc2 = nn.Linear(self.h[1], self.h[0])
#         self.decoder_fc1 = nn.Linear(self.h[0], input_dim)

#         if bn:
#             self.dbn1 = nn.BatchNorm1d(self.h[1])
#             self.dbn2 = nn.BatchNorm1d(self.h[0])

#     # ----- Forward -----
#     def encoder(self, x):
#         x = self.activation(self.encoder_fc1(x))
#         if self.bn: x = self.bn1(x)

#         x = self.activation(self.encoder_fc2(x))
#         if self.bn: x = self.bn2(x)

#         y = self.activation(self.encoder_fc3(x))
#         if self.bn: y = self.bn3(y)

#         return y  # final embedding (same as y_flat in RSRAE)

#     def decoder(self, y):
#         z = self.activation(self.decoder_fc3(y))
#         if self.bn: z = self.dbn1(z)

#         z = self.activation(self.decoder_fc2(z))
#         if self.bn: z = self.dbn2(z)

#         x_hat = self.decoder_fc1(z)
#         return x_hat

#     def forward(self, x):
#         y = self.encoder(x)      # same dimension as RSRAE: hidden_layer_sizes[2]
#         x_hat = self.decoder(y)  # reconstruction
#         return y, x_hat

#     # ----- Loss -----
#     def reconstruction_error(self, x, x_hat):
#         if self.loss_norm_type.lower() in ["mse", "f", "frob"]:
#             return torch.mean(torch.norm(x - x_hat, dim=1) ** 2)
#         elif self.loss_norm_type.lower() == "l1":
#             return torch.mean(torch.norm(x - x_hat, p=1, dim=1))
#         else:
#             return torch.mean(torch.norm(x - x_hat, dim=1))

#     # ----- Training -----
#     def fit(self, X, batch_size=128, device="cuda", x_val=None):
#         self.to(device)
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

#         n_samples = X.shape[0]
#         n_batch = (n_samples - 1) // batch_size + 1

#         for epoch in range(self.epoch_size):
#             idx = np.random.permutation(n_samples)

#             for b in range(n_batch):
#                 start = b * batch_size
#                 end = min((b + 1) * batch_size, n_samples)

#                 batch = X[idx[start:end]]

#                 optimizer.zero_grad()
#                 _, x_hat = self.forward(batch)
#                 loss = self.reconstruction_error(batch, x_hat)
#                 loss.backward()
#                 optimizer.step()

#             if self.batch_show and (epoch + 1) % self.batch_show == 0:
#                 print(f"Epoch {epoch+1}/{self.epoch_size} - Loss: {loss.item():.4f}")

#     def get_latent(self, X, device="cuda"):
#         self.to(device)
#         with torch.no_grad():
#             y, _ = self.forward(X)
#         return y.cpu().numpy()

#     def get_output(self, X, device="cuda"):
#         self.to(device)
#         with torch.no_grad():
#             _, x_hat = self.forward(X)
#         return x_hat.cpu().numpy()



# class AE_BASELINE(BaselineModel):
#     def __init__(self, args):
#         self.model = AE_NoLatent(**args)

#     def train(self, X_train, device='cuda'):
#         self.model.fit(X_train, 128, None, device)
#         return self.model

#     def test(self, X_test, y_test, device='cuda'):
#         with torch.no_grad():
#             features = self.model.get_output(X_test, device=device)
#             flat_output = np.reshape(features, (features.shape[0], -1))
#             flat_input = np.reshape(X_test.cpu().numpy(), (X_test.shape[0], -1))

#             cosine_similarity = np.sum(flat_output * flat_input, -1) / (
#                 np.linalg.norm(flat_output, axis=-1) + 1e-6) / (
#                 np.linalg.norm(flat_input, axis=-1) + 1e-6)

#             auc, fpr95, ap = ev.evaluation(y_test, -cosine_similarity, verbose=False)
#             return auc, fpr95, ap
