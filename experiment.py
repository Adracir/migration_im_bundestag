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


# 17.305317401885986 seconds taken for testepoch
def prepared_corpus_to_freq_dict(corpus_name):
    corpus_unflattened = utils.unpickle(corpus_name)
    wordlist = list(itertools.chain(*corpus_unflattened))
    return Counter(wordlist)


def find_frequencies_for_keywords(corpus_name):
    freq_dict = prepared_corpus_to_freq_dict(corpus_name)
    df = load_keywords()
    keywords = df['word'].tolist()
    return {k: freq_dict.get(k, 0) for k in keywords}


start = time.time()
print(find_frequencies_for_keywords('data/corpus/prepared_testepoch'))
end = time.time()
print(f'{end-start} seconds taken')

# {'Flüchtling': 2388, 'Vertriebener': 0, 'Heimatvertriebener': 57, 'Fremdarbeiter': 10, 'Gastarbeiter': 5,
# 'Ausländer': 378, 'Migrant': 0, 'Asylant': 0, 'Scheinasylant': 0, 'Wirtschaftsasylant': 0, 'Wirtschaftsflüchtling': 0,
# 'Asylmissbrauch': 0, 'Asylbewerber': 0, 'Asylsuchender': 0, 'Asylberechtigter': 0, 'Multikulturell': 0,
# 'Multinational': 0, 'Integration': 597, 'Einwanderungsland': 0, 'Migrationshintergrund': 0, 'Geflüchteter': 0,
# 'Flüchtender': 0}
