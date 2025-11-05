from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import torch


def build_vocab(corpus, min_freq=2):

    counter = Counter(word for text in corpus for word in text.split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def initialize_context_vectors(net, train_loader):
    """
    Initialize the context vectors from an initial run of k-means++ on simple average sentence embeddings

    Returns
    -------
    centers : ndarray, [n_clusters, n_features]
    """

    # Get vector representations
    X = ()
    for data in train_loader:
        inputs, _, _, _ = data
        # text.shape = (sentence_length, batch_size)

        X_batch = net.pretrained_model(inputs)
        # X_batch.shape = (sentence_length, batch_size, embedding_size)

        # compute mean and normalize
        X_batch = torch.mean(X_batch, dim=0)
        X_batch = X_batch / torch.norm(X_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
        X_batch[torch.isnan(X_batch)] = 0
        # X_batch.shape = (batch_size, embedding_size)

        X += (X_batch.cpu().data.numpy(),)

    X = np.concatenate(X)
    n_attention_heads = net.n_attention_heads

    kmeans = KMeans(n_clusters=n_attention_heads).fit(X)
    centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)


    return centers


def cvdd_model_pipeline(data_train, data_test, attention_size, n_attention_heads, embedding_type, seq_len, batch_size, shuffle, tokenizer=None, vocab=None):

    from Data_Preparation.Dataset.ADdatasets import CVDDDatasetWrapper
    from Modelisation.Baselines.CVDD.networks import embedding_layer, cvdd_Net
    from torch.utils.data import DataLoader
    # ================================
    # ------------ BERT --------------
    # ================================
    if embedding_type == 'bert':
        if tokenizer is not None:
            cvdd_dataset_train = CVDDDatasetWrapper(data_train, embedding_type='bert', tokenizer=tokenizer, seq_len=seq_len)
            cvdd_dataset_test = CVDDDatasetWrapper(data_test, embedding_type='bert', tokenizer=tokenizer, seq_len=seq_len)
            pretrained_model = embedding_layer.EmbeddingFactory.create('bert', bert_name='distilbert-base-uncased', trainable=True)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameters 'bert_name' and 'tokenizer' is required")

    # ================================
    # ----------- GLOVE --------------
    # ================================
    elif embedding_type == 'glove': 
        if vocab is not None:
            cvdd_dataset_train = CVDDDatasetWrapper(data_train, embedding_type='glove', vocab=vocab, seq_len=seq_len)
            cvdd_dataset_test = CVDDDatasetWrapper(data_test, embedding_type='glove', vocab=vocab, seq_len=seq_len)
            pretrained_model = embedding_layer.EmbeddingFactory.create('glove',
                                    glove_path='./Modelisation/Baselines/CVDD/embedding_models/glove.6B.300d.txt',
                                    vocab=vocab,
                                    embedding_dim=300,
                                    trainable=True)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameter 'vocab' is required")
        
    # ================================
    # ----------- FASTTEXT -----------
    # ================================
    elif embedding_type == 'fasttext':
        if vocab is not None:
            cvdd_dataset_train = CVDDDatasetWrapper(data_train, embedding_type='fasttext', vocab=vocab, seq_len=seq_len)   
            cvdd_dataset_test = CVDDDatasetWrapper(data_test, embedding_type='fasttext', vocab=vocab, seq_len=seq_len)   
            pretrained_model = embedding_layer.EmbeddingFactory.create('fasttext',
                                    fasttext_path='./Modelisation/Baselines/CVDD/embedding_models/wiki-news-300d-1M.vec',
                                    vocab=vocab,
                                    embedding_dim=300,
                                    trainable=True)
        else:
            raise Exception(f"when 'embedding_type' = '{embedding_type}', the parameter 'vocab' is required")
        
    else: raise Exception(f" the 'embedding_type' {embedding_type} is not possible with CVDD, please choose ('bert','glove','fasttext')")
        

    dl_train = DataLoader(cvdd_dataset_train, batch_size=batch_size, shuffle=shuffle)
    dl_test = DataLoader(cvdd_dataset_test, batch_size=batch_size, shuffle=False)
    
    model = cvdd_Net.CVDDNet(pretrained_model, attention_size, n_attention_heads)

    return model, dl_train, dl_test
