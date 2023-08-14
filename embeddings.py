import gensim
from gensim.models import Word2Vec, KeyedVectors
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import calc
import prepare_corpus
import pickle
import utils


def make_word_emb_model(data, sg=1, vec_dim=100, window=5):
    """
    initialize and train a Word2Vec model with gensim from the given data
    :param data: list of lists, containing tokenized words in tokenized sentences
    (as can be generated from raw text with preprocess_text_for_word_embedding_creation(filename))
    :param sg: if 0, method CBOW is used, if 1, Skipgram
    :param vec_dim: defines the dimensions of the resulting vectors
    :param window: defines the window that is used to create the embeddings
    """
    print('\nGenerating Model')
    return gensim.models.Word2Vec(data, min_count=1, sg=sg, vector_size=vec_dim, window=window)


def evaluate_embeddings(keyed_vectors, evaluation_set='222'):
    """
    print the evaluation of a given model in correlation (Pearson and Spearman) to a human-based list of
    words (based on Gurevych)
    For a well-functioning model, the first value is expected to be as high as possible, the pvalue is expected to be
    less than 0.05
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param evaluation_set: shortcut for evaluation set to be used, either '65', '222', or '350'"""
    df = pd.read_csv(f'data/evaluation/wortpaare{evaluation_set}.gold.pos.txt', delimiter=':')
    gold_standard_relatedness = [float(x) for x in df['GOLDSTANDARD'].tolist()]
    words1 = df['#WORD1'].tolist()
    words2 = df['WORD2'].tolist()
    embedding_relatedness = []
    unuseable_indices_list = []
    for i in range(0, len(words1)):
        try:
            embedding_relatedness.append(keyed_vectors.similarity(words1[i], words2[i]))
        except KeyError:
            unuseable_indices_list.append(i)
            print(f'word {words1[i]} or {words2[i]} not in list')
    for u in sorted(unuseable_indices_list, reverse=True):
        del gold_standard_relatedness[u]
    print(pearsonr(gold_standard_relatedness, embedding_relatedness))
    print(spearmanr(gold_standard_relatedness, embedding_relatedness))


# prepare and save epoch corpus from txt
'''data = prepare_corpus.prepare_text_for_embedding_training('data/corpus/epoch3.txt', False)
with open("data/corpus/epoch3_prepared_nolemma", "wb") as fp:   #Pickling
    pickle.dump(data, fp)'''
# load epoch corpus and create model
unpickled = utils.unpickle("data/corpus/epoch3_prepared_lemma")
model = make_word_emb_model(unpickled, sg=0, vec_dim=300, window=7)
# save model
word_vectors = model.wv
word_vectors.save('data/models/epoch3_lemma_300d_7w_cbow.wordvectors')
print('Model generated. Evaluating.')
# load existing model
'''
word_vectors = KeyedVectors.load('data/models/epoch1_lemma_100d_3w_skipgram.wordvectors')'''
# evaluate word embeddings
evaluate_embeddings(word_vectors, evaluation_set='222')

# epoch1_prepared_lemma
# 96 Wörter nicht da
# PearsonRResult(statistic=0.12440467960688481, pvalue=0.16688283570164342)
# SpearmanrResult(correlation=0.16931448717091163, pvalue=0.05907587116500668)
# epoch2_prepared_lemma
# 82 Wörter nicht da
# PearsonRResult(statistic=0.16622818164974834, pvalue=0.05049664692556891)
# SpearmanrResult(correlation=0.18028305039707626, pvalue=0.03369143082567974)
# epoch3_prepared_lemma
# 76 Wörter nicht da
# PearsonRResult(statistic=0.11321474192537147, pvalue=0.17514683510913576)
# SpearmanrResult(correlation=0.13499589086894043, pvalue=0.10546534181124151)
# epoch4_prepared_lemma
# 67 Wörter nicht da
# PearsonRResult(statistic=0.21147468823687632, pvalue=0.008468522546265076)
# SpearmanrResult(correlation=0.22178729811927836, pvalue=0.005702780927131208)
# epoch5_prepared_lemma
# 68 Wörter nicht da
# PearsonRResult(statistic=0.13254725775879847, pvalue=0.10240835410799254)
# SpearmanrResult(correlation=0.15411646931285033, pvalue=0.05716305238378497)
# epoch6_prepared_lemma
# 60 Wörter nicht da
# PearsonRResult(statistic=0.15444703501732426, pvalue=0.05044190007230016)
# SpearmanrResult(correlation=0.15121806581710784, pvalue=0.05551575231165192)
# epoch7_prepared_lemma
# 71 Wörter nicht da
# PearsonRResult(statistic=0.14485495257686024, pvalue=0.07696020582008638)
# SpearmanrResult(correlation=0.12362426216573527, pvalue=0.13175849318981225)
# epoch8_prepared_lemma:
# 80 Wörter nicht da
# PearsonRResult(statistic=0.07715726418503396, pvalue=0.36314284888090576)
# SpearmanrResult(correlation=0.09230683939108082, pvalue=0.2763072931852034)
# epoch8_prepared_nolemma:
# 87 Wörter nicht da
# PearsonRResult(statistic=0.06089268221323073, pvalue=0.48459645240382593)
# SpearmanrResult(correlation=0.08499477109418159, pvalue=0.328848349017106)

