import pandas as pd

import prepare_corpus
import embeddings
import experiment
import visualizations
import utils
from gensim.models import KeyedVectors
from collections import Counter
import math
import itertools


if __name__ == '__main__':
    # epoch_id = 8
    # unite text for epochs
    '''prepare_corpus.pure_text_to_epoch_txt(epoch_id)'''
    # preprocess text and save this step
    '''data = prepare_corpus.prepare_text_for_embedding_training(f"data/corpus/epoch{epoch_id}.txt", lemmatize=True)
    utils.make_pickle(f"data/corpus/epoch{epoch_id}_prepared_lemma", data)'''
    # use preprocessed text to train and save cbow model
    '''prepared_corpus = utils.unpickle(f"data/corpus/epoch{epoch_id}_prepared_lemma")
    model = embeddings.make_word_emb_model(prepared_corpus, sg=0, vec_dim=200, window=7)
    # save model
    print("Saving model")
    model.save(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.model')
    # save word vectors
    word_vectors = model.wv
    word_vectors.save(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.wordvectors')'''
    # count frequencies and save in csv
    '''experiment.save_frequency_info_in_csv()'''
    # get most frequent word & its occurrence for each epoch
    '''for epoch in range(1, 9):
        wv = KeyedVectors.load(f'data/models/base_models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors')
        most_frequent_word = wv.index_to_key[0]
        wordlist = experiment.prepared_corpus_to_wordlist(f"data/corpus/epoch{epoch}_prepared_lemma")
        dict = Counter(wordlist)
        print(f"Epoch {epoch}: most frequent word '{most_frequent_word}' appears {dict.get(most_frequent_word, 0)} times!")'''
    # analyze first occurrences of keywords and save in keywords csv
    '''experiment.create_kw_occurrences_and_merge_to_keyword_list()'''
    # plot corpus frequency distribution
    '''visualizations.plot_frequency_distribution_for_corpora()'''
    '''visualizations.plot_frequency_distribution_for_corpora_keywords()'''
    # plot frequencies
    '''for absolute in [True, False]:
        visualizations.plot_frequencies(absolute=absolute)'''
    '''visualizations.plot_comparing_frequencies()'''
    # analyse sentiment of words
    '''for sentiword_set, with_axis in itertools.product(['', 'religion', 'political', 'race', 'ethics'], [True, False]):
        experiment.analyse_senti_valuation_of_keywords(sentiword_set=sentiword_set, with_axis=True)'''
    # normalize with_axis values afterwards
    '''experiment.normalize_with_axis_senti_and_save_to_csv()'''
    # plot valuations
    '''for with_axis, absolute in itertools.product([True, False], repeat=2):
        visualizations.plot_sentiments(['standard', 'political'], with_axis=with_axis, absolute=absolute)'''
    # plot comparing valuations
    # for sentiword_set, with_axis in itertools.product(['standard', 'political'], [True, False]):
    '''for sentiword_set in ['standard', 'political']:
        visualizations.plot_comparing_sentiments(sentiword_set, with_axis=True)'''
    # save nearest neighbors without alignment
    '''experiment.save_nearest_neighbors()'''
    # possibility to analyse some words more closely, see their context in the corpus
    '''prepare_corpus.print_contexts_for_word_from_lemmatized_corpus('motzen', 1)'''
    # align models to make visualization of nearest neighbors over time
    '''embeddings.align_according_to_occurrences()'''
    # add model for epoch 8 for word "multikulturell"
    # TODO: maybe avoid or include in align_according_to_occurrences?
    '''embeddings.align_two_models(7, 8, 4)'''
    # save nearest neighbors from aligned models to csv
    '''experiment.save_nearest_neighbors(aligned=True)'''
    # TODO: compare nearest neighbors from plot with those from csvs
    # TODO: try with different parameters till it looks good. also with doubles
    visualizations.plot_tsne_according_to_occurrences(k=10, perplexity=5, keep_doubles=True, iter=5000)
    '''visualizations.plot_words_from_time_epochs_tsne(epochs=[3, 4, 5, 6, 7, 8], target_word='Asylant',
                                                    aligned_base_folder='data/models/aligned_models/start_epoch_3',
                                                    k=10, perplexity=10, keep_doubles=False, iter=5000)'''
    # save distance for each keyword between the aligned epochs
    '''experiment.calculate_cosine_development_for_each_keyword()'''
    # plot these distances
    '''visualizations.plot_cosine_development_each_word()'''
    # plot similarities between certain groups of words over the years
    '''visualizations.plot_exemplary_comparisons()'''
    # upgrade expected_values.csv with translation for senti
    # TODO: rather use symbols +, - etc?
    '''df = pd.read_csv('data/expected_values.csv')
    translation_df = pd.read_csv('data/expected_senti_translation.csv')
    expected_senti = df['expected_valuation'].tolist()
    expected_senti_written = []
    for f in expected_senti:
        written = translation_df[translation_df['senti_value'] == f]['written'].iloc[0]
        expected_senti_written.append(written)
    df['written_senti'] = expected_senti_written
    df.to_csv('data/expected_values.csv')'''
