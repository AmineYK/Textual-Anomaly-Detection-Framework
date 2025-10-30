import torch.nn as nn
from abc import ABC, abstractmethod
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer




class BaseEmbeddingEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.embedding_size = None

    @abstractmethod
    def forward(self, x):
        pass


class BERTEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, bert_name='distilbert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.model = AutoModel.from_pretrained(bert_name)
        self.embedding_size = self.model.config.hidden_size

    def forward(self, texts):
        # texts: list[str]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        return hidden_states.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
    



class GloVeEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, glove_path, vocab, embedding_dim=300):
        super().__init__()
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(len(vocab), embedding_dim)

        # Charger les poids GloVe
        weights_matrix = np.zeros((len(vocab), embedding_dim))
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab:
                    vector = np.asarray(values[1:], dtype='float32')
                    weights_matrix[vocab[word]] = vector
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

    def forward(self, x):
        # x: (sentence_length, batch_size)
        embedded = self.embedding(x)
        return embedded  # (sentence_length, batch_size, embedding_size)
    

# class FastTextEmbeddingEncoder(BaseEmbeddingEncoder):
#     def __init__(self, fasttext_path, vocab, embedding_dim=300):
#         super().__init__()
#         self.model = fasttext.load_model(fasttext_path)
#         self.embedding_size = embedding_dim
#         self.embedding = nn.Embedding(len(vocab), embedding_dim)

#         weights_matrix = np.zeros((len(vocab), embedding_dim))
#         for word, idx in vocab.items():
#             weights_matrix[idx] = self.model.get_word_vector(word)
#         self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

#     def forward(self, x):
#         return self.embedding(x)



class FastTextVecEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, fasttext_path, vocab, trainable=False):
        """
        Args:
            vec_path: chemin vers le fichier .vec FastText
            vocab: dictionnaire {mot: index}
            trainable: bool, si True on peut fine-tuner les embeddings
        """
        super().__init__()
        self.vocab = vocab

        # Initialisation aléatoire des poids
        vocab_size = len(vocab)
        embedding_dim = None
        weights_matrix = None

        print(f"Loading FastText .vec embeddings from {fasttext_path} ...")
        with open(fasttext_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip().split()
            if len(first_line) == 2:
                # Première ligne = header (vocab_size, dim)
                embedding_dim = int(first_line[1])
            else:
                # Pas de header
                embedding_dim = len(first_line) - 1
                # remettre la première ligne dans le parsing
                f.seek(0)

            weights_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))

            for line in f:
                parts = line.rstrip().split(' ')
                word = parts[0]
                vec = np.asarray(parts[1:], dtype='float32')
                if word in vocab:
                    idx = vocab[word]
                    weights_matrix[idx] = vec

        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = trainable

    def forward(self, x):
        return self.embedding(x)


    

class TFIDFEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, corpus, max_features=5000):
        super().__init__()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.vectorizer.fit(corpus)
        self.embedding_size = len(self.vectorizer.get_feature_names_out())

    def forward(self, texts):
        tfidf_vectors = self.vectorizer.transform(texts).toarray()
        tensor = torch.tensor(tfidf_vectors, dtype=torch.float32)
        # TF-IDF n’a pas de structure séquentielle, donc ici shape = (batch_size, embedding_size)
        return tensor.unsqueeze(0)  # (1, batch_size, embedding_size)


class EmbeddingFactory:

    @staticmethod
    def create(embedding_type: str, **kwargs):
        
        embedding_type = embedding_type.lower()

        if embedding_type == 'bert':
            return BERTEmbeddingEncoder(
                bert_name=kwargs.get('bert_name', 'distilbert-base-uncased')
            )

        elif embedding_type == 'glove':
            return GloVeEmbeddingEncoder(
                glove_path=kwargs['glove_path'],
                vocab=kwargs['vocab'],
                embedding_dim=kwargs.get('embedding_dim', 300)
            )

        elif embedding_type == 'fasttext':
            return FastTextVecEmbeddingEncoder(
                fasttext_path=kwargs['fasttext_path'],
                vocab=kwargs['vocab'],
                trainable=kwargs.get('trainable', False)
            )

        elif embedding_type == 'tfidf':
            return TFIDFEmbeddingEncoder(
                corpus=kwargs['corpus'],
                max_features=kwargs.get('max_features', 5000)
            )

        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

