import os.path

import gensim
from gensim.models import Word2Vec, KeyedVectors
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import prepare_corpus
import pickle
import utils
import numpy as np


def make_word_emb_model(data, sg=1, vec_dim=100, window=5):
    """
    initialize and train a Word2Vec model with gensim from the given data
    :param data: list of lists, containing tokenized words in tokenized sentences
    (as can be generated from raw text with preprocess_text_for_word_embedding_creation(filename))
    :param sg: if 0, method CBOW is used, if 1, Skipgram
    :param vec_dim: defines the dimensions of the resulting vectors
    :param window: defines the window that is used to create the embeddings
    """
    print('Generating Model')
    return gensim.models.Word2Vec(data, sg=sg, vector_size=vec_dim, window=window)


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


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """ Src: https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.

    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # re-filling the normed vectors, see comment by amacanovic in
    # https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8
    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)

    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """ src: https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key

        print(len(m.wv.key_to_index), len(m.wv.vectors))

    return (m1, m2)


def align_two_models(epoch1, epoch2, start_epoch, loophole='0'):
    base_path = 'data/models/base_models/'
    aligned_base_path = f'data/models/aligned_models/start_epoch_{start_epoch}{f"_lh_{loophole}" if not str(0) in loophole else ""}/'
    if epoch1 == start_epoch:
        model1_path = f'{base_path}epoch{epoch1}_lemma_200d_7w_cbow.model'
    else:
        model1_path = f'{aligned_base_path}epoch{epoch1}_lemma_200d_7w_cbow_aligned.model'
    model2_path = f'{base_path}epoch{epoch2}_lemma_200d_7w_cbow.model'
    model1 = Word2Vec.load(model1_path)
    model2 = Word2Vec.load(model2_path)
    smart_procrustes_align_gensim(model1, model2)
    # save maybe updated version of models
    if epoch1 == start_epoch:
        model1.save(f'{aligned_base_path}epoch{epoch1}_lemma_200d_7w_cbow_aligned.model')
        model1_aligned_vectors = model1.wv
        model1_aligned_vectors.save(f'{aligned_base_path}epoch{epoch1}_lemma_200d_7w_cbow_aligned.wordvectors')
    model2.save(f'{aligned_base_path}epoch{epoch2}_lemma_200d_7w_cbow_aligned.model')
    model2_vectors = model2.wv
    model2_vectors.save(f'{aligned_base_path}epoch{epoch2}_lemma_200d_7w_cbow_aligned.wordvectors')


def evaluate_aligned_models(epoch1, epoch2, start_epoch, loophole='0'):
    base_path = 'data/models/base_models/'
    aligned_base_path = f'data/models/aligned_models/start_epoch_{start_epoch}{f"_lh_{loophole}" if not str(0) in loophole else ""}/'
    start_model = Word2Vec.load(f'{aligned_base_path}epoch{start_epoch}_lemma_200d_7w_cbow_aligned.model')
    model1_aligned = Word2Vec.load(f'{aligned_base_path}epoch{epoch1}_lemma_200d_7w_cbow_aligned.model')
    model2 = Word2Vec.load(f'{base_path}epoch{epoch2}_lemma_200d_7w_cbow.model')
    model2_aligned = Word2Vec.load(f'{aligned_base_path}epoch{epoch2}_lemma_200d_7w_cbow_aligned.model')
    # Nearest Neighbors gleich je Epoche? Ja, abgesehen von Kürzungen des Vokabulars durchs Alignment
    print(f'model{epoch2}: \n{model2.wv.most_similar(positive="Flüchtling")}\n')
    print(f'model{epoch2}_aligned: \n{model2_aligned.wv.most_similar(positive="Flüchtling")}\n')
    # Ähnlichkeit zwischen 1_2 & 2_2 messbar? Jaa! Auch eine rückwirkende Ähnlichkeit zur ersten Epoche lässt sich bei
    # sequentiellem Alignment feststellen!
    print(f'sim {epoch1}_aligned & {epoch2}: {utils.similarity(model1_aligned.wv["der"], model2.wv["der"])}')
    print(f'sim {epoch1}_aligned & {epoch2}_aligned: {utils.similarity(model1_aligned.wv["der"], model2_aligned.wv["der"])}')
    print(f'sim {start_epoch} & {epoch2}_aligned: {utils.similarity(start_model.wv["der"], model2_aligned.wv["der"])}')


# manually checking if everything that is needed is there is required!
def align_according_to_occurrences():
    # iterate rows in keywords_merged.csv
    df = pd.read_csv('data/keywords_merged.csv')
    for index, row in df.iterrows():
        # if word never or only once occurs, ignore
        if row.first_occ_epoch != 0 and row.last_occ_epoch - (len(row.loophole.split('_')) if int(row.loophole) != 0 else 0) - row.first_occ_epoch > 0:
            # check if resp. start_epoch-folder exists
            aligned_base_folder = f'data/models/aligned_models/start_epoch_{row.first_occ_epoch}{f"_lh_{row.loophole}" if not str(0) in row.loophole else ""}'
            if os.path.isdir(aligned_base_folder):
                # check if it contains all (and only the) necessary models (iterate)
                for epoch in range(row.first_occ_epoch, row.last_occ_epoch + 1):
                    epoch_aligned_model_path = f'{aligned_base_folder}/epoch{epoch}_lemma_200d_7w_cbow_aligned.model'
                    if str(epoch) not in row.loophole:
                        if not os.path.exists(epoch_aligned_model_path):
                            print(f'WARNING: Epoch {epoch} missing in correct folder {aligned_base_folder}! Keyword {row.keyword} cannot be evaluated')
                    else:
                        if os.path.exists(epoch_aligned_model_path):
                            print(f'WARNING: Epoch {epoch} falsely existing in folder {aligned_base_folder}! Keyword {row.keyword} cannot be evaluated')
            # if not existing, create new start_epoch-folder with evtl. resp. hole
            else:
                os.makedirs(aligned_base_folder)
                # iterate all needed models and align accordingly
                necessary_epochs = [item for item in range(row.first_occ_epoch, row.last_occ_epoch + 1) if str(item) not in row.loophole]
                print(f'aligning epochs {", ".join(map(str, necessary_epochs))}')
                for epoch1 in necessary_epochs:
                    if epoch1 != necessary_epochs[-1]:
                        epoch2 = necessary_epochs[necessary_epochs.index(epoch1) + 1]
                        align_two_models(epoch1, epoch2, row.first_occ_epoch, row.loophole)
                        evaluate_aligned_models(epoch1, epoch2, row.first_occ_epoch, row.loophole)


# prepare and save epoch corpus from txt
'''data = prepare_corpus.prepare_text_for_embedding_training('data/corpus/epoch3.txt', False)
with open("data/corpus/epoch3_prepared_nolemma", "wb") as fp:   #Pickling
    pickle.dump(data, fp)'''
# load epoch corpus and create model
'''unpickled = utils.unpickle("data/corpus/epoch8_prepared_lemma")
model = make_word_emb_model(unpickled, sg=0, vec_dim=200, window=7)
# save model
model.save('data/models/epoch8_lemma_200d_7w_cbow.model')
# save word vectors
word_vectors = model.wv
word_vectors.save('data/models/epoch8_lemma_200d_7w_cbow.wordvectors')'''
# print('Model generated. Evaluating.')
# load existing keyedvectors
# word_vectors = KeyedVectors.load('data/models/aligned_1_2_zhicongchen.wordvectors')
# evaluate word embeddings
# evaluate_embeddings(word_vectors, evaluation_set='222')
# print(f"Most similar to Flüchtling: \n200d_7w_cbow: {word_vectors.most_similar(positive='Flüchtling')}")
# align_according_to_occurrences()
# load two Models and try to align them
# align_two_models(3, 4, 3)
# load changed models after alignment and check if successful
# evaluate_aligned_models(3, 4, 3)


# epoch 1
# Most similar to Flüchtling:
# 100d_7w_skipgram: [('Vertriebene', 0.9247472286224365), ('Aussiedler', 0.8800923824310303), ('Sowjetzonenflüchtling', 0.8710741996765137), ('Heimatvertrieben', 0.8563693761825562), ('Umsiedler', 0.8511750102043152), ('Zuwanderer', 0.8078914284706116), ('Evakuierte', 0.8012499213218689), ('kriegsgeschädigt', 0.7985544800758362), ('kriegssachgeschädigt', 0.7906972765922546), ('Spätaussiedler', 0.7858254313468933)]
# 100d_3w_skipgram: [('Vertriebene', 0.8831362128257751), ('Sowjetzonenflüchtling', 0.8685519695281982), ('Umsiedler', 0.8674185276031494), ('Aussiedler', 0.8635145425796509), ('Heimatvertrieben', 0.8486348986625671), ('Spätheimkehrer', 0.8190304040908813), ('Evakuierte', 0.8149763345718384), ('Ausgebombte', 0.8101011514663696), ('Zuwanderer', 0.8075731992721558), ('Heimkehrer', 0.8025177121162415)]
# 100d_5w_skipgram: [('Vertriebene', 0.9234288930892944), ('Heimatvertrieben', 0.8803684115409851), ('Sowjetzonenflüchtling', 0.8616733551025391), ('Umsiedler', 0.8553743362426758), ('Aussiedler', 0.8488149642944336), ('Evakuierte', 0.840194821357727), ('Zuwanderer', 0.8017981648445129), ('kriegsgeschädigt', 0.798314094543457), ('Kriegssachgeschädigte', 0.7959287762641907), ('Spätaussiedler', 0.7893429398536682)]
# 300d_7w_cbow: [('Sowjetzonenflüchtling', 0.7900823354721069), ('Aussiedler', 0.7854887247085571), ('Umsiedler', 0.7768185138702393), ('Vertriebene', 0.7615658044815063), ('Evakuierte', 0.758834183216095), ('Heimatvertrieben', 0.7527918219566345), ('Ausgebombte', 0.7226887941360474), ('Zuwanderer', 0.7015560865402222), ('Spätheimkehrer', 0.6965314149856567), ('Einheimische', 0.6745473742485046)]
# 200d_7w_cbow: [('Sowjetzonenflüchtling', 0.8131846189498901), ('Aussiedler', 0.7983640432357788), ('Vertriebene', 0.7973310947418213), ('Evakuierte', 0.7799867987632751), ('Heimatvertrieben', 0.7708706855773926), ('Ausgebombte', 0.7627128958702087), ('Umsiedler', 0.7624684572219849), ('Spätheimkehrer', 0.7375816702842712), ('Zuwanderer', 0.726215660572052), ('Heimkehrer', 0.7070479393005371)]
# 100d_7w_cbow: [('Aussiedler', 0.8655480742454529), ('Sowjetzonenflüchtling', 0.8509020805358887), ('Vertriebene', 0.8355391621589661), ('Evakuierte', 0.8339203596115112), ('Umsiedler', 0.8236814141273499), ('Heimatvertrieben', 0.8204135298728943), ('Ausgebombte', 0.8091865181922913), ('Zuwanderer', 0.7807300090789795), ('Bombengeschädigte', 0.7643333673477173), ('Spätheimkehrer', 0.7606095671653748)]
# 100d_5w_cbow: [('Aussiedler', 0.8376753330230713), ('Heimatvertrieben', 0.8355817198753357), ('Vertriebene', 0.8155630230903625), ('Umsiedler', 0.811207115650177), ('Evakuierte', 0.8098737597465515), ('Sowjetzonenflüchtling', 0.7993268370628357), ('Ausgebombte', 0.7879809737205505), ('Spätheimkehrer', 0.7593058347702026), ('Zuwanderer', 0.7569756507873535), ('Heimkehrer', 0.7488828897476196)]

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

