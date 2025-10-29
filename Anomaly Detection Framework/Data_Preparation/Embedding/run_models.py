from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# to create file 'glove_300d.kv'

# glove_input_file = 'glove.6B.300d.txt'
# word2vec_output_file = 'glove.6B.300d.word2vec.txt'
# glove2word2vec(glove_input_file, word2vec_output_file)

# glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# glove_model.save(".-./emb_models/glove_300d.kv")


# to create 'fasttext_300d.kv' 

# fasttext_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)

# # Sauvegarder au format .kv pour un chargement plus rapide
# fasttext_model.save("../emb_models/fasttext_300d.kv")