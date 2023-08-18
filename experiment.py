import gensim
from gensim.models import KeyedVectors
import pandas as pd
import nltk
import utils
import itertools
import time
from collections import Counter

# TODO: visualize frequency of words, maybe like this:

"""fdist = nltk.FreqDist(nouns)

pprint(fdist.most_common(10))
fdist.plot(50,cumulative=False)"""

# TODO: nearest neighbors
# TODO: alignment & diachronic visualizations
# TODO: positive/negative aspect of words. maybe use SentiWord lists of esp. positive and negative words and measure
#  distance/similarity


def load_model(name):
    return KeyedVectors.load(f"data/models/{name}")


def load_keywords():
    df = pd.read_csv('data/keywords.csv')
    return df


def prepared_corpus_to_count_dict(corpus_name):
    corpus_unflattened = utils.unpickle(corpus_name)
    # flatten corpus to one layered list
    wordlist = list(itertools.chain(*corpus_unflattened))
    return Counter(wordlist)


def find_counts_for_keywords(epoch):
    count_dict = prepared_corpus_to_count_dict(f"data/corpus/epoch{epoch}_prepared_lemma")
    df = load_keywords()
    keywords = df['word'].tolist()
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


start = time.time()
save_frequency_info_in_csv(1)
end = time.time()
print(f'{end-start} seconds taken')
