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