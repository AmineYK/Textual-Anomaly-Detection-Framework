import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Modelisation.Baselines.baseline import BaselineModel
import Modelisation.evaluation as ev

class MultiContextSelfAttention(nn.Module):
    def __init__(self, embed_dim, latent_dim, num_contexts):

        super().__init__()
        self.num_contexts = num_contexts

        self.W1 = nn.Linear(embed_dim, latent_dim, bias=False)  
        self.W2 = nn.Linear(latent_dim, num_contexts, bias=False)  

    
    def forward(self, x):
        
        # (batch, 1, di)
        x = x.unsqueeze(1)  
        
        #(batch, 1, num_con)    
        x1 = torch.tanh(self.W1(x))
        A = torch.softmax(self.W2(x1), dim=1)
        # A = torch.softmax(self.att(x), dim=1)

        # (batch, num_con, dim)
        # somme pondérée des embeddings des tokens pour chaque contexte
        Z = torch.einsum("btk,btd->bkd", A, x) 
        return Z
    
class CVDD(nn.Module):
    def __init__(self, embed_dim, latent_dim=150, num_contexts=8):

        super().__init__()
        self.attention = MultiContextSelfAttention(embed_dim, latent_dim, num_contexts)
        self.context_vectors = nn.Parameter(torch.randn(num_contexts, embed_dim))
    
    def forward(self, x):
        z = self.attention(x)
        return z

def cvdd_loss(z, context_vectors, lambda_div=0.1):
    c = context_vectors.unsqueeze(0)
    
    dist = torch.norm(z - c, dim=-1)
    loss_compact = torch.mean(torch.min(dist, dim=1)[0])
    
    C = context_vectors
    gram = torch.matmul(C, C.T)
    loss_div = torch.sum(torch.abs(gram - torch.eye(C.size(0), device=C.device)))
    
    return loss_compact + lambda_div * loss_div

@torch.no_grad()
def compute_scores(model, emb, device):
    model.eval()
    
    emb = emb.to(device)
    z   = model(emb)
    c   = model.context_vectors.unsqueeze(0)
    
    dist= torch.norm(z - c, dim=-1)
    return torch.min(dist, dim=1)[0].cpu().numpy()

class CVDDModel(BaselineModel):
    def __init__(self, args):
        
        self.n_attention_heads = args['n_attention_heads']
        self.latent_dim = args['latent_dim']
        self.lr = args['lr']
        self.lambda_p = args['lambda_p']
        self.n_epochs = args['n_epochs']
        self.batch_size = args['batch_size']
        self.device = args['device']

    def train(self, data_train):

        train_loader = DataLoader(TensorDataset(torch.tensor(data_train['sbert_embeddings'])), batch_size=self.batch_size, shuffle=True)
        dim = next(iter(train_loader))[0].shape[1]

        self.model = CVDD(embed_dim=dim, latent_dim=self.latent_dim, num_contexts=self.n_attention_heads).to(self.device)
        opt   = optim.Adam(self.model.parameters(), lr=self.lr)
        nb_epochs = self.n_epochs
        
        for epoch in range(nb_epochs):
            total_loss = 0
            for x in train_loader:
                x = x[0].to(self.device)
                z = self.model(x)
                
                loss = cvdd_loss(z, self.model.context_vectors, self.lambda_p)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
            if (epoch % (nb_epochs//3) == 0):print(f" Epoch {epoch+1} — Loss: {total_loss/len(train_loader):.4f}")


        return self.model

    def test(self, data_test):

        test_scores = compute_scores(self.model, Tensor(data_test['sbert_embeddings']), self.device)

        auc, ap, fpr95 = ev.evaluation(data_test['anomaly_class'], test_scores, verbose=False)

        return auc, fpr95, ap