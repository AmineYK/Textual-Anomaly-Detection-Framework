from Modelisation.Baselines.CVDD.networks.cvdd_Net import *
from Modelisation.Baselines.CVDD.networks.process import *
from Modelisation.Baselines.CVDD.networks.self_attention import *


hidden_size = 150
attention_size = 250
n_attention_heads = 2

SA = SelfAttention(hidden_size=hidden_size, 
              attention_size=attention_size, 
              n_attention_heads=n_attention_heads)

print(SA)
