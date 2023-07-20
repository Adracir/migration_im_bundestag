import gensim
from gensim.models import Word2Vec, KeyedVectors
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import calc
import prepare_corpus
import pickle


def make_word_emb_model(data, sg=1, vec_dim=100):
    """
    initialize and train a Word2Vec model with gensim from the given data
    :param data: list of lists, containing tokenized words in tokenized sentences
    (as can be generated from raw text with preprocess_text_for_word_embedding_creation(filename))
    :param sg: if 0, method CBOW is used, if 1, Skipgram
    :param vec_dim: defines the dimensions of the resulting vectors
    """
    print('\nGenerating Model')
    return gensim.models.Word2Vec(data, min_count=1, sg=sg, vector_size=vec_dim, window=5)


def evaluate_embeddings(keyed_vectors, distance_measure='cosine'):
    """
    print the evaluation of a given model in correlation (Pearson and Spearman) to a human-based list of
    words (based on Gurevych)
    For a well-functioning model, the first value is expected to be as high as possible, the pvalue is expected to be
    less than 0.05
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param distance_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used similarity/distance measure
    """
    df = pd.read_csv('data/evaluation/wortpaare222.gold.pos.txt', delimiter=':')
    gold_standard_relatedness = [float(x) for x in df['GOLDSTANDARD'].tolist()]
    words1 = df['#WORD1'].tolist()
    words2 = df['WORD2'].tolist()
    embedding_relatedness = []
    unuseable_indices_list = []
    for i in range(0, len(words1)):
        try:
            vec_word1 = keyed_vectors[words1[i]]
            vec_word2 = keyed_vectors[words2[i]]
            embedding_relatedness.append(calc.distance(vec_word1, vec_word2, distance_measure))
        except KeyError:
            unuseable_indices_list.append(i)
            print(f'word {words1[i]} or {words2[i]} not in list')
    embedding_relatedness = calc.normalize_and_reverse_distances(embedding_relatedness, distance_measure)
    for u in sorted(unuseable_indices_list, reverse=True):
        del gold_standard_relatedness[u]
    print(pearsonr(gold_standard_relatedness, embedding_relatedness))
    print(spearmanr(gold_standard_relatedness, embedding_relatedness))


'''data = prepare_corpus.prepare_text_for_embedding_training('data/corpus/epoch8.txt', True)
with open("data/corpus/epoch8_prepared_lemma", "wb") as fp:   #Pickling
    pickle.dump(data, fp)'''
with open("data/corpus/epoch8_prepared_nolemma", "rb") as fp:  # Unpickling
    unpickled = pickle.load(fp)
model = make_word_emb_model(unpickled)
print('Model generated. Evaluating.')
evaluate_embeddings(model.wv)
# epoch8_prepared_nolemma:
# 87 Wörter nicht da
# PearsonRResult(statistic=0.06089268221323073, pvalue=0.48459645240382593)
# SpearmanrResult(correlation=0.08499477109418159, pvalue=0.328848349017106)
# epoch8_prepared_lemma:
# 80 Wörter nicht da
# PearsonRResult(statistic=0.07715726418503396, pvalue=0.36314284888090576)
# SpearmanrResult(correlation=0.09230683939108082, pvalue=0.2763072931852034)

