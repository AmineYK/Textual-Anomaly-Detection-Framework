import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer




class EmbeddingEncoder:
    def __init__(self, model=None, model_name=None, type_emd='glove', device='cuda'):
        
        if model is not None and type_emd == 'glove':
            self.model = GloVeEmbeddingEncoder(model, device)
            
        elif model is not None and type_emd == 'fasttext':
            self.model = FastTextEmbeddingEncoder(model, device)
                              
        elif model is not None and type_emd == 'tfidf':
            self.model = TFIDFEmbeddingEncoder(model) 
            
        elif model_name is not None and type_emd == 'bert':
            self.model = BERTEmbeddingEncoder(model_name, device) 
      
        else : raise Exception ("'model' & 'model_name' are None type, at least one is requered")
        
        
    def forward(self, dataloader):
        
        return self.model.forward(dataloader)
    
    
    
class TFIDFEmbeddingEncoder:
    def __init__(self, tfidf_vectorizer):
        
        self.vectorizer = tfidf_vectorizer
        self.fitted = False

    def forward(self, dataloader):

        dataset = dataloader.dataset
        texts = dataset['text']
        
        if not self.fitted:
            vectors = self.vectorizer.fit_transform(texts)
            self.fitted = True
        else:
            vectors = self.vectorizer.transform(texts)
        
        vectors = vectors.toarray()
        dataset = dataset.add_column("tfidf_embedding", list(vectors))
        
        return DataLoader(dataset, batch_size=dataloader.batch_size)


class BERTEmbeddingEncoder:
    def __init__(self, model_name, device):

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def forward(self, dataloader):
        
        def add_col(example, col_name, embedding):
            example[col_name] = embedding
            return example

        dataset = dataloader.dataset
        texts = dataset['text']
        
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
        
        dataset = dataset.map(add_col,fn_kwargs={"col_name": 'bert_cls', "embedding": cls_emb.cpu().numpy()} )
        dataset = dataset.map(add_col,fn_kwargs={"col_name": 'bert_embedding', "embedding": last_hidden.cpu().numpy()} )
        dataset = dataset.map(add_col,fn_kwargs={"col_name": 'bert_embedding_mean', "embedding": mean_emb.cpu().numpy()} )
        
        return DataLoader(dataset, batch_size = dataloader.batch_size)
    
    
class GloVeEmbeddingEncoder:
    def __init__(self, model, device):

        self.device = device 
        self.model = model
        self.embedding_dim = self.model.vector_size
        
        
    def forward(self, dataloader):
        
        def add_col(example, col_name, embedding):
            example[col_name] = np.array(embedding)
            return example
        
        dataset = dataloader.dataset
        texts = dataset['text']
        
        vectors = []
        for text in texts:
            words = [w for w in text.split() if w in self.model.key_to_index]
            if words:
                emb = torch.tensor([self.model[w] for w in words]).mean(dim=0)
            else:
                emb = torch.zeros(self.embedding_dim)
            vectors.append(emb.cpu().numpy())
        
        # dataset = dataset.map(add_col,fn_kwargs={"col_name": 'glove_embedding', "embedding": emb} )
        # vectors = torch.stack(vectors).cpu().numpy()
        dataset = dataset.add_column("glove_embedding",vectors) 

        return DataLoader(dataset, batch_size = dataloader.batch_size)
    
    
class FastTextEmbeddingEncoder:
    def __init__(self, model, device):

        self.device = device
        self.model = model
        self.embedding_dim = self.model.vector_size
        
        
    def forward(self, dataloader):
        
        def add_col(example, col_name, embedding):
            example[col_name] = embedding
            return example
        
        dataset = dataloader.dataset
        texts = dataset['text']
        
        vectors = []
        for text in texts:
            words = [w for w in text.split() if w in self.model.key_to_index]
            if words:
                emb = torch.tensor([self.model[w] for w in words]).mean(dim=0)
            else:
                emb = torch.zeros(self.embedding_dim)
            vectors.append(emb.cpu().numpy())
        
        # vectors = torch.stack(vectors).cpu().numpy()
        # dataset = dataset.map(add_col,fn_kwargs={"col_name": 'fasttext_embedding', "embedding": vectors} )
        
        dataset = dataset.add_column("fasttext_embedding",vectors) 

        return DataLoader(dataset, batch_size = dataloader.batch_size)

     
