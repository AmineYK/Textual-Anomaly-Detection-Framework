import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any
from torch.utils.data import DataLoader
import numpy as np


# BERT Enmbedding Extractor Class
#--------------------------------

class BERTEmbeddingExtractor:
    def __init__(self, model_name, device):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_batch(self, texts) -> Dict[str, Any]:
      
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state  
        cls_emb = last_hidden[:, 0, :]           
        mean_emb = last_hidden.mean(dim=1)       

        
        return {
            "CLS": cls_emb.cpu().numpy(),
            "embedding": last_hidden.cpu().numpy(),
            "embedding_mean": mean_emb.cpu().numpy()
        }


# Extract Embedding
#------------------

def extract_embeddings(dataset, extractor, batch_size = 32):

    def process(batch):

        emb = extractor.encode_batch(batch["text"])
        
        # here it's a fusion of the dicts
        batch.update(emb)
      
        return batch

    return DataLoader(dataset.dataset.map(
        process,
        batched=True,
        batch_size=batch_size
    ), batch_size=batch_size, shuffle=True)  
    
     
