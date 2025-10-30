from collections import Counter

def build_vocab(corpus, min_freq=2):
    counter = Counter(word for text in corpus for word in text.split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab