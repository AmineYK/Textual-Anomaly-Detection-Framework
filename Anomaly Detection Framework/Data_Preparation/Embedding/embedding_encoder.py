import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from abc import ABC, abstractmethod

import os


################################################
################## ENCODER  ####################
################################################

class EmbeddingEncoder:
    def __init__(self, model_name=None, type_emd='glove'):
        
        if type_emd == 'glove':
            self.model = GloVeEmbeddingEncoder(model_name)
            
        elif type_emd == 'fasttext':
            self.model = FastTextEmbeddingEncoder(model_name)
                              
        elif type_emd == 'tfidf':
            self.model = TFIDFEmbeddingEncoder(model_name) 
            
        elif type_emd == 'bert':
            self.model = BERTEmbeddingEncoder(model_name) 
      
        else : raise Exception ("'model' & 'model_name' are None type, at least one is requered")
        
        
    def forward(self, dataset):
        
        return self.model.forward(dataset)
    

################################################
################## ABSTRACT ####################
################################################

class BaseEmbeddingEncoder(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name

    @abstractmethod
    def forward(self, dataset):
        pass
    

################################################
################### TDFIDF  ####################
################################################
    
class TFIDFEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self,model_name):
    
        super().__init__(model_name)
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      
            min_df=3,               
            max_df=0.8,            
            ngram_range=(1, 2),      
            stop_words='english',    
            lowercase=True,          
            norm='l2',               
            use_idf=True,            
            smooth_idf=True,         
            sublinear_tf=True        
        )
        self.fitted = False

    def forward(self, dataset):

        texts = dataset['text']
        
        if not self.fitted:
            vectors = self.vectorizer.fit_transform(texts)
            self.fitted = True
        else:
            vectors = self.vectorizer.transform(texts)
        
        vectors = vectors.toarray()
        dataset = dataset.add_column("tfidf_embedding", list(vectors))
        
        return dataset


    
################################################
#################### BERT  #####################
################################################    

class BERTEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, model_name):

        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def forward(self, dataset):
        
        def add_col(example, col_name, embedding):
            example[col_name] = embedding
            return example
        
        texts = dataset['text']
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
      
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state  
        cls_emb = last_hidden[:, 0, :]           
        mean_emb = last_hidden.mean(dim=1)   
        
        dataset = dataset.map(add_col,fn_kwargs={"col_name": 'bert_cls', "embedding": cls_emb.cpu().numpy()} )
        dataset = dataset.map(add_col,fn_kwargs={"col_name": 'bert_embedding', "embedding": last_hidden.cpu().numpy()} )
        dataset = dataset.map(add_col,fn_kwargs={"col_name": 'bert_embedding_mean', "embedding": mean_emb.cpu().numpy()} )
        
        return dataset
    
    
################################################
#################### GloVe  ####################
################################################
    
class GloVeEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, model_name):

        super().__init__(model_name)
        self.model_name = model_name

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '..', 'emb_models', self.model_name)
        self.model = KeyedVectors.load(model_path, mmap='r')

        # self.model = KeyedVectors.load(f"Data_Preparation/emb_models/{self.model_name}", mmap='r')
        self.embedding_dim = self.model.vector_size
        
        
    def forward(self, dataset):
        
        def add_col(example, col_name, embedding):
            example[col_name] = np.array(embedding)
            return example
        
        texts = dataset['text']
        
        vectors = []
        for text in texts:
            words = [w for w in text.split() if w in self.model.key_to_index]
            if words:
                emb = torch.tensor(np.array([self.model[w] for w in words])).mean(dim=0)
            else:
                emb = torch.zeros(self.embedding_dim)
            vectors.append(emb.cpu().numpy())
        
        # dataset = dataset.map(add_col,fn_kwargs={"col_name": 'glove_embedding', "embedding": emb} )
        # vectors = torch.stack(vectors).cpu().numpy()
        dataset = dataset.add_column("glove_embedding",vectors) 

        return dataset
  

###################################################
#################### FastText  ####################
###################################################

    
class FastTextEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, model_name):
        
        super().__init__(model_name)
        self.model_name = model_name

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '..', 'emb_models', self.model_name)
        self.model = KeyedVectors.load(model_path, mmap='r')

        # self.model = KeyedVectors.load(f"Data_Preparation/emb_models/{self.model_name}", mmap='r')
        self.embedding_dim = self.model.vector_size
        
        
    def forward(self, dataset):
        
        def add_col(example, col_name, embedding):
            example[col_name] = embedding
            return example
        
        texts = dataset['text']
        
        vectors = []
        for text in texts:
            words = [w for w in text.split() if w in self.model.key_to_index]
            if words:
                emb = torch.tensor(np.array([self.model[w] for w in words])).mean(dim=0)
            else:
                emb = torch.zeros(self.embedding_dim)
            vectors.append(emb.cpu().numpy())
        
        # vectors = torch.stack(vectors).cpu().numpy()
        # dataset = dataset.map(add_col,fn_kwargs={"col_name": 'fasttext_embedding', "embedding": vectors} )
        
        dataset = dataset.add_column("fasttext_embedding",vectors) 

        return dataset

     
