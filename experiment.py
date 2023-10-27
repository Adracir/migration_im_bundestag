import os.path

import utils
import itertools
import time
from collections import Counter
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

from scipy.spatial.distance import cosine


# TODO: nearest neighbors
# TODO: alignment & diachronic visualizations
# TODO: positive/negative aspect of words. maybe use SentiWord lists of esp. positive and negative words and measure
#  distance/similarity


def prepared_corpus_to_wordlist(corpus_name):
    corpus_unflattened = utils.unpickle(corpus_name)
    # flatten corpus to one layered list
    wordlist = list(itertools.chain(*corpus_unflattened))
    return wordlist


def find_counts_for_keywords(count_dict):
    keywords = utils.load_keywords()
    return {k: count_dict.get(k, 0) for k in keywords}


# TODO: avoid double executions, most of this is already done in prepared_corpus_to_count_dict
'''def count_total_words_in_epoch_corpus(epoch):
    corpus_unflattened = utils.unpickle(f"data/corpus/epoch{epoch}_prepared_lemma")
    wordlist = list(itertools.chain(*corpus_unflattened))
    return len(wordlist)'''


def save_frequency_info_in_csv():
    utils.write_info_to_csv('data/results/freqs.csv', ['epoch', 'keyword', 'count', 'freq'])
    for epoch in range(1, 9):
        print(f'epoch {epoch}: simplifying corpus for counting operations')
        wordlist = prepared_corpus_to_wordlist(f"data/corpus/epoch{epoch}_prepared_lemma")
        dict = Counter(wordlist)
        count_dict = find_counts_for_keywords(dict)
        total_words = len(wordlist)
        print('iterating keywords and saving information')
        for kw in count_dict.keys():
            count = count_dict[kw]
            freq = count/total_words
            utils.write_info_to_csv('data/results/freqs.csv', [epoch, kw, count, freq], 'a')


def find_first_occurrences_for_keywords():
    # load keywords
    keywords = utils.load_keywords()
    # load freqs
    df = pd.read_csv('data/results/freqs.csv')
    utils.write_info_to_csv("data/results/kw_occurrences.csv", ['keyword', 'first_occ_epoch', 'last_occ_epoch',
                                                                'loophole', 'low_occ'])
    # for each keyword find first & last non-null freq-epoch
    for kw in keywords:
        kw_freqs_df = df[df['keyword'] == kw]
        # freq infos for kw df
        non_zero_epochs_df = kw_freqs_df[kw_freqs_df['count'] != 0]
        # für welche Epochs ist count != 0?
        non_zero_epochs = non_zero_epochs_df['epoch'].tolist()
        low_occ_epochs_df = kw_freqs_df[kw_freqs_df['freq'] < 1e-06]
        low_occ_epochs = low_occ_epochs_df['epoch'].tolist()
        first_occ_epoch = min(non_zero_epochs) if non_zero_epochs else 0
        last_occ_epoch = max(non_zero_epochs) if non_zero_epochs else 0
        loopholes = []
        relevant_low_occ_epochs = []
        for i in range(first_occ_epoch, last_occ_epoch+1):
            if i not in non_zero_epochs:
                loopholes.append(f'{i}')
                print(f"WARNING: {kw} has a loophole at {i}")
            if i in low_occ_epochs:
                relevant_low_occ_epochs.append(f'{i}')
        # write into csv: keyword, first_occ_epoch, last_occ_epoch
        utils.write_info_to_csv("data/results/kw_occurrences.csv", [kw, first_occ_epoch, last_occ_epoch,
                                                                    '_'.join(loopholes) if len(loopholes) > 0 else 0,
                                                                    '_'.join(relevant_low_occ_epochs) if
                                                                    len(relevant_low_occ_epochs) > 0 else 0], mode='a')


def create_kw_occurrences_and_merge_to_keyword_list():
    find_first_occurrences_for_keywords()
    df1 = pd.read_csv('data/keywords.csv')
    df2 = pd.read_csv('data/results/kw_occurrences.csv')
    merged_df = df1.merge(df2, on='keyword', how='outer')
    merged_df.to_csv('data/keywords_merged.csv', index=False)


# TODO: maybe ignore words in epoch that appear seldom, e.g. less than 1e-06
def save_nearest_neighbors(aligned=False):
    if aligned:
        df = pd.read_csv('data/keywords_merged.csv')
    epochs = range(1, 9)
    base_folder = 'data/models/base_models'
    # load keywords from csv
    keywords = utils.load_keywords()
    # initialize csv to save results to
    words_similarities_headings = []
    for no in range(1, 11):
        words_similarities_headings.append(f'Word_{no}')
        words_similarities_headings.append(f'Similarity_{no}')
    save_file_path = f'data/results/nearest_neighbors{"_aligned" if aligned else ""}.csv'
    utils.write_info_to_csv(save_file_path, ['Keyword', 'Epoch'] + words_similarities_headings)
    # iterate keywords
    for kw in keywords:
        if aligned:
            row = df[df['keyword'] == kw].iloc[0]
            base_folder = f'data/models/aligned_models/start_epoch_{row.first_occ_epoch}{f"_lh_{row.loophole}" if not str(0) in row.loophole else ""}'
            epochs = [item for item in range(row.first_occ_epoch, row.last_occ_epoch + 1) if
                                str(item) not in row.loophole]
        # iterate base models/epochs
        for epoch in epochs:
            word_vectors_path = f'{base_folder}/epoch{epoch}_lemma_200d_7w_cbow{"_aligned" if aligned else ""}.wordvectors'
            word_vectors = KeyedVectors.load(word_vectors_path)
            # retrieve 10 nearest neighbors of kw
            try:
                nearest_neighbors = word_vectors.most_similar(positive=kw)
                # format nearest neighbors
                words_similarities = []
                for n in nearest_neighbors:
                    words_similarities.append(n[0])
                    words_similarities.append(n[1])
                # save in csv file
                utils.write_info_to_csv(save_file_path, [kw, epoch] + words_similarities, mode='a')
            except KeyError:
                print(f"Keyerror: Key '{kw}' not present in vocabulary for epoch {epoch}")


def comparing_connotations(model1, model2, word, k=10, verbose=True):
    """ copied from https://gensim.narkive.com/ZsBAPGm4/11863-word2vec-alignment
    calculates a relative semantic shift between a word in two different models
    - `model1`, `model2`: Are gensim word2vec models. KeyedVectors
    - `word` is a string representation of a given word.
    - `k` is the size of the word's neighborhood (# of its closest words in its
    vector space).
    """
    # Check that this word is present in both models
    # if not word in model1.wv.vocab or not word in model2.wv.vocab:
    if not word in model1.key_to_index or not word in model2.key_to_index:
        print("!! Word %s not present in both models." % word)
        return None
    # Get the two neighborhoods
    neighborhood1 = [w for w, c in model1.most_similar(word, topn=k)]
    neighborhood2 = [w for w, c in model2.most_similar(word, topn=k)]
    # Print?
    if verbose:
        print('>> Neighborhood of associations of the word "%s" in model1:' % word)
        print(', '.join(neighborhood1))
        print('>> Neighborhood of associations of the word "%s" in model2:' % word)
        print(', '.join(neighborhood2))
    # Get the 'meta' neighborhood (both combined)
    meta_neighborhood = list(set(neighborhood1) | set(neighborhood2))
    # Filter the meta neighborhood so that it contains only words present in both models
    meta_neighborhood = [w for w in meta_neighborhood if w in model1.key_to_index and w in model2.key_to_index]
    # For both models, get a similarity vector between the focus word and all of the words in the meta neighborhood
    vector1 = [model1.similarity(word, w) for w in meta_neighborhood]
    vector2 = [model2.similarity(word, w) for w in meta_neighborhood]
    # Compute the cosine distance *between* those similarity vectors
    dist = cosine(vector1, vector2)
    # Return this cosine distance -- a measure of the relative semantic shift for this word between these two models
    return dist


def compare_connotations_for_all_keywords():
    # load keywords and epochs
    df = pd.read_csv('data/keywords_merged.csv')
    keywords = df['keyword'].tolist()
    epochs_df = pd.read_csv('data/epochs.csv')
    # prepare output csv
    output_csv_path = 'data/results/compared_connotations.csv'
    utils.write_info_to_csv(output_csv_path, ['keyword', 'first_epoch', 'next_epoch', 'epoch_range_str', 'distance'])
    # depending on occurrences, get necessary models/epochs
    for kw in keywords:
        row = df[df['keyword'] == kw].iloc[0]
        necessary_epochs = [item for item in range(row.first_occ_epoch, row.last_occ_epoch + 1) if str(item) not in row.loophole]
        aligned_base_folder = f'data/models/aligned_models/start_epoch_{row.first_occ_epoch}{f"_lh_{row.loophole}" if not str(0) in row.loophole else ""}'
        # iterate the epochs/models, always taking two neighboring ones at once
        for epoch in necessary_epochs:
            if epoch == necessary_epochs[-1]:
                continue
            else:
                next_epoch = necessary_epochs[necessary_epochs.index(epoch)+1]
                wordvectors1 = KeyedVectors.load(f'{aligned_base_folder}/epoch{epoch}_lemma_200d_7w_cbow_aligned.wordvectors')
                wordvectors2 = KeyedVectors.load(f'{aligned_base_folder}/epoch{next_epoch}_lemma_200d_7w_cbow_aligned.wordvectors')
                # get distance and save into csv
                vector1 = wordvectors1[kw]
                vector2 = wordvectors2[kw]
                dist = cosine(vector1, vector2)
                # TODO: test this alternative with unaligned vectors
                # dist = comparing_connotations(wordvectors1, wordvectors2, kw)
                epoch_row = epochs_df[epochs_df['epoch_id'] == epoch].iloc[0]
                next_epoch_row = epochs_df[epochs_df['epoch_id'] == next_epoch].iloc[0]
                epoch_range_str = f'{epoch_row.written_form_short} bis {next_epoch_row.written_form_short}'
                utils.write_info_to_csv(output_csv_path, [kw, epoch, next_epoch, epoch_range_str, dist], mode='a')


            # TODO: maybe also calculate and plot distances for pairs or triples of words, as in Braun


# adapted from https://github.com/leahannah/weat_demo/blob/main/weat.py
def s(w, A, B, word_vectors):
    """Calculate bias score of attribute word w and two target sets A and B
    :param w: attribute word
    :param A: target set
    :param B: other target set
    :param word_vectors: keyedvectors to use"""
    cos_wa = [word_vectors.similarity(w, a) for a in A if a in word_vectors.index_to_key]
    cos_wb = [word_vectors.similarity(w, b) for b in B if b in word_vectors.index_to_key]
    return np.mean(cos_wa) - np.mean(cos_wb)


def analyse_senti_valuation_of_keywords(sentiword_model=""):
    print("preparing sentiment analysis")
    # load keywords
    keywords = utils.load_keywords()
    # load sentiwords according to choice
    senti_file_path = f"data/{sentiword_model}{'_' if sentiword_model else ''}sentiwords.csv"
    df = pd.read_csv(senti_file_path)
    # group sentiwords by value (A: +1/B: -1)
    df_pos = df[df["value"] == 1]
    df_neg = df[df["value"] == -1]
    pos_words = df_pos["word"].tolist()
    neg_words = df_neg["word"].tolist()
    # prepare output csv
    output_file_path = "data/results/senti.csv"
    if not os.path.exists(output_file_path):
        utils.write_info_to_csv(output_file_path, ["word", "epoch", "sentiword_model", "value"])
    # iterate keywords
    for kw in keywords:
        print(f"Analysing key word {kw}")
        # for keyword: iterate epochs
        # TODO: maybe get epochs programmatically??
        for epoch in range(1, 9):
            # get associated wordvectors
            word_vectors = KeyedVectors.load(f"data/models/base_models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors")
            try:
                # calculate bias value of word with s()
                senti = s(kw, pos_words, neg_words, word_vectors)
                # save value in csv
                utils.write_info_to_csv(output_file_path, [kw, epoch, sentiword_model if sentiword_model else "standard", senti], mode="a")
            except KeyError:
                print(f"Keyword {kw} not in vocabulary of epoch {epoch}! Omitted from analysis.")


# print(comparing_connotations(KeyedVectors.load('data/models/aligned_models/start_epoch_1/epoch1_lemma_200d_7w_cbow_aligned.wordvectors'),
                    #    KeyedVectors.load('data/models/aligned_models/start_epoch_1/epoch2_lemma_200d_7w_cbow_aligned.wordvectors'),
                    #    'Flüchtling'))
