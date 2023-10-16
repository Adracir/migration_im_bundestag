import utils
import itertools
import time
from collections import Counter
from gensim.models import KeyedVectors
import pandas as pd


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


# TODO: maybe ignore words in epoch that appear seldom, e.g. less than 1e-06
def save_nearest_neighbors():
    # load keywords from csv
    keywords = utils.load_keywords()
    # initialize csv to save results to
    words_similarities_headings = []
    for no in range(1, 11):
        words_similarities_headings.append(f'Word_{no}')
        words_similarities_headings.append(f'Similarity_{no}')
    save_file_path = 'data/results/nearest_neighbors.csv'
    utils.write_info_to_csv(save_file_path, ['Keyword', 'Epoch'] + words_similarities_headings)
    # iterate keywords
    for kw in keywords:
        # iterate base models/epochs
        for epoch in range(1, 9):
            word_vectors_path = f'data/models/base_models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors'
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


# TODO: migrate and try
def comparing_connotations(model1, model2, word, k=10, verbose=True):
    """ copied from https://gensim.narkive.com/ZsBAPGm4/11863-word2vec-alignment
    calculates a relative semantic shift between a word in two different models
    - `model1`, `model2`: Are gensim word2vec models. KeyedVectors
    - `word` is a string representation of a given word.
    - `k` is the size of the word's neighborhood (# of its closest words in its
    vector space).
    """
    # Import function for cosine distance
    from scipy.spatial.distance import cosine
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


# start = time.time()
# for epoch in range(1, 9):
# save_frequency_info_in_csv(epoch)
# end = time.time()
# print(f'{end-start} seconds taken')
# word_vectors1 = KeyedVectors.load('data/models/epoch1_lemma_200d_7w_cbow.wordvectors')
# word_vectors2 = KeyedVectors.load('data/models/epoch2_lemma_200d_7w_cbow.wordvectors')
# print(comparing_connotations(word_vectors1, word_vectors2, "Flüchtling"))
# save_nearest_neighbors()
# find_first_occurrences_for_keywords()
