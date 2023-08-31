import utils
import itertools
import time
from collections import Counter
from gensim.models import KeyedVectors

# TODO: visualize frequency of words, maybe like this:

"""fdist = nltk.FreqDist(nouns)

pprint(fdist.most_common(10))
fdist.plot(50,cumulative=False)"""

# TODO: nearest neighbors
# TODO: alignment & diachronic visualizations
# TODO: positive/negative aspect of words. maybe use SentiWord lists of esp. positive and negative words and measure
#  distance/similarity


def prepared_corpus_to_count_dict(corpus_name):
    corpus_unflattened = utils.unpickle(corpus_name)
    # flatten corpus to one layered list
    wordlist = list(itertools.chain(*corpus_unflattened))
    return Counter(wordlist)


def find_counts_for_keywords(epoch):
    count_dict = prepared_corpus_to_count_dict(f"data/corpus/epoch{epoch}_prepared_lemma")
    keywords = utils.load_keywords()
    return {k: count_dict.get(k, 0) for k in keywords}


# TODO: avoid double executions, most of this is already done in prepared_corpus_to_count_dict
def count_total_words_in_epoch_corpus(epoch):
    corpus_unflattened = utils.unpickle(f"data/corpus/epoch{epoch}_prepared_lemma")
    wordlist = list(itertools.chain(*corpus_unflattened))
    return len(wordlist)


def save_frequency_info_in_csv(epoch):
    count_dict = find_counts_for_keywords(epoch)
    total_words = count_total_words_in_epoch_corpus(epoch)
    for kw in count_dict.keys():
        count = count_dict[kw]
        freq = count/total_words
        utils.write_info_to_csv('data/results/freqs.csv', [epoch, kw, count, freq], 'a')
# TODO: use freqs csv to update keywords csv with first time occurrences of each word


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
word_vectors1 = KeyedVectors.load('data/models/epoch1_lemma_200d_7w_cbow.wordvectors')
word_vectors2 = KeyedVectors.load('data/models/epoch2_lemma_200d_7w_cbow.wordvectors')
print(comparing_connotations(word_vectors1, word_vectors2, "Flüchtling"))