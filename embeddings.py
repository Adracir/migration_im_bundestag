import os.path

import gensim
from gensim.models import Word2Vec
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np

import utils


# Embedding Training
def make_word_emb_model(data, sg=1, vec_dim=100, window=5):
    """
    initialize and train a Word2Vec model with gensim from the given data
    :param data: nested list, containing tokenized words in tokenized sentences
    (as can be generated from raw text with prepare_corpus.prepare_text_for_embedding_training(filename))
    :param sg: if 0, method CBOW is used, if 1, Skipgram
    :param vec_dim: defines the dimensions of the resulting vectors
    :param window: defines the window that is used to create the embeddings
    """
    print('Generating Model')
    return gensim.models.Word2Vec(data, sg=sg, vector_size=vec_dim, window=window)


def evaluate_embeddings(keyed_vectors, evaluation_set='222'):
    """
    print the evaluation of a given model in correlation (Pearson and Spearman) to a human-based list of words
    (Gurevych, Iryna 2005. German Relatedness Datasets.
    https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2440 [Stand 2023-07-13].)
    For a well-functioning model, the first value is expected to be as high as possible, the pvalue is expected to be
    less than 0.05
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param evaluation_set: shortcut for evaluation set to be used, either '65', '222', or '350'"""
    # read gold standard relatedness, saved in file
    df = pd.read_csv(f'data/evaluation/wortpaare{evaluation_set}.gold.pos.txt', delimiter=':')
    gold_standard_relatedness = [float(x) for x in df['GOLDSTANDARD'].tolist()]
    # measure embedding relatedness
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
    # remove unavailable words from gold standard as well
    for u in sorted(unuseable_indices_list, reverse=True):
        del gold_standard_relatedness[u]
    # print stats
    print(pearsonr(gold_standard_relatedness, embedding_relatedness))
    print(spearmanr(gold_standard_relatedness, embedding_relatedness))


# Alignment
def align_according_to_occurrences():
    """
    check which alignments are needed according to the occurrences of the keywords and align the respective models
    can also be used to check if all needed aligned models are available
    :return: (create folders giving the start & end epoch as well as loopholes, if an epoch should be skipped
    and save the aligned models inside. Also print warnings if important alignments are missing)
    """
    # iterate rows in keywords_merged.csv
    df = pd.read_csv('data/keywords_merged.csv')
    for index, row in df.iterrows():
        # if word occurs never or only once, ignore
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
                necessary_epochs = utils.get_necessary_epochs_for_kw(row)
                print(f'aligning epochs {", ".join(map(str, necessary_epochs))}')
                for epoch1 in necessary_epochs:
                    if epoch1 != necessary_epochs[-1]:
                        epoch2 = necessary_epochs[necessary_epochs.index(epoch1) + 1]
                        align_two_models(epoch1, epoch2, row.first_occ_epoch, row.loophole)


def align_two_models(epoch1, epoch2, start_epoch, loophole='0'):
    """
    align two models using smart procrustes algorithm
    :param epoch1: first epoch, if not equal to start epoch, already aligned with the previous ones
    :param epoch2: epoch to be aligned
    :param start_epoch: epoch in which the process had started, depending on occurrence of keyword that should be
    analyzed using these models
    :param loophole: epoch that should be skipped in the whole alignment process for the respective keyword
    :return: (save to the respective folder, depending on start_epoch and loopholes)
    """
    base_path = 'data/models/base_models/'
    aligned_base_path = f'data/models/aligned_models/start_epoch_{start_epoch}' \
                        f'{f"_lh_{loophole}" if not str(0) in loophole else ""}/'
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


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """ Src: https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8
    Based on original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
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
