
from networks.embedding_layer import *
from networks.cvdd_Net import *
from networks.utils import *



def main():
    hidden_size = 150
    attention_size = 250
    n_attention_heads = 2

    # SA = self_attention.SelfAttention(hidden_size=hidden_size, 
    #               attention_size=attention_size, 
    #               n_attention_heads=n_attention_heads)

    # print(SA)

    # cvdd = cvdd_Net.CVDDNet(SA, attention_size, n_attention_heads)
    # print(cvdd)

    corpus = ["the cat sat on the mat", "dogs bark loudly"]

    # pretrained_model = EmbeddingFactory.create(
    #     'tfidf',
    #     corpus=corpus,
    #     max_features=1000
    # )
    # model = CVDDNet(pretrained_model)
    # print(model)

    vocab = build_vocab(corpus,min_freq=1)
    print(vocab)

    # pretrained_model = EmbeddingFactory.create(
    # 'glove',
    # glove_path='./embedding_models/glove.6B.300d.txt',
    # vocab=vocab,
    # embedding_dim=300
    # )
    # model = CVDDNet(pretrained_model)
    # print(model)


    pretrained_model = EmbeddingFactory.create(
        'fasttext',
        fasttext_path='./embedding_models/wiki-news-300d-1M.vec',
        vocab=vocab,
        trainable=True
    )
    model = CVDDNet(pretrained_model)
    print(model)


    # pretrained_model = embedding_layer.EmbeddingFactory.create('bert', bert_name='distilbert-base-uncased')
    # model = CVDDNet(pretrained_model)
    # print(model)

if __name__ == "__main__":
    main()