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
import numpy as np


if __name__ == '__main__':
    epoch_id = 8
    # unite text for epochs
    '''prepare_corpus.pure_text_to_epoch_txt(epoch_id)'''
    # preprocess text and save this step
    '''data = prepare_corpus.prepare_text_for_embedding_training(f"data/corpus/epoch{epoch_id}.txt", lemmatize=True)
    utils.make_pickle(f"data/corpus/epoch{epoch_id}_prepared_lemma", data)'''
    # count frequencies and save in csv
    '''experiment.save_frequency_info_in_csv()'''
    # analyze first occurrences of keywords and save in keywords csv
    '''experiment.create_kw_occurrences_and_merge_to_keyword_list()'''
    # append expected.csv with written form to enable plotting
    '''experiment.include_written_form_in_expected_csv(method='freq')'''
    # save pMW slices of freqs results to enable comparing with "freq classes"
    '''experiment.make_freq_slices()'''
    # plot frequencies
    '''for absolute in [True, False]:
        visualizations.plot_frequencies(absolute=absolute, adapted_to_results=True)'''
    '''visualizations.plot_frequencies()'''
    '''visualizations.plot_comparing_frequencies()'''
    # calculate and plot mean frequencies
    '''experiment.calculate_mean_frequency_over_all_epochs()'''
    '''visualizations.plot_mean_frequencies_as_bar_plot()'''
    # use preprocessed text to train and save cbow models
    '''for epoch_id in range(1, 9):
        prepared_corpus = utils.unpickle(f"data/corpus/epoch{epoch_id}_prepared_lemma")
        model = embeddings.make_word_emb_model(prepared_corpus, sg=0, vec_dim=200, window=7)
        # save model
        print("Saving model")
        model.save(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.model')
        # save word vectors
        word_vectors = model.wv
        word_vectors.save(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.wordvectors')'''
    # optional: evaluate models
    '''for epoch_id in range(1, 9):
        word_vectors = KeyedVectors.load(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.wordvectors')
        print(f'epoch: {epoch_id}, vocabsize: {len(word_vectors)}')'''
    # analyse sentiment of words
    '''for sentiword_set in ['', 'political', 'combination']:
        experiment.analyse_senti_valuation_of_keywords(sentiword_set=sentiword_set)'''
    # append expected.csv with written form to enable plotting
    '''experiment.include_written_form_in_expected_csv(method='senti')'''
    # save senti slices of senti results for plots
    '''experiment.make_senti_slices()'''
    # plot senti
    '''visualizations.plot_sentiments(['combination'], absolute=False, adapted_to_results=True)'''
    '''visualizations.plot_comparing_sentiments()'''
    # calculate and plot mean sentiment over all time
    '''experiment.calculate_mean_sentiment_over_all_epochs()'''
    '''visualizations.plot_mean_sentiments_as_bar_plot()'''
    # save nearest neighbors without alignment
    '''experiment.save_nearest_neighbors()'''
    # possibility to analyse some words more closely, see their context in the corpus
    '''prepare_corpus.print_contexts_for_word_from_lemmatized_corpus('multikulturell', 4)'''
    # align models to make visualization of nearest neighbors over time
    '''embeddings.align_according_to_occurrences()'''
    # add some missing aligned model for epoch 8 for words "multikulturell" & "Migrationshintergrund".
    #  manually checked if adding these would be valid for the resp. folders, following warnings from process before
    '''embeddings.align_two_models(5, 6, 4)
    embeddings.align_two_models(6, 7, 4)
    embeddings.align_two_models(7, 8, 4)
    embeddings.align_two_models(7, 8, 6)'''
    # save nearest neighbors from aligned models to csv
    '''experiment.save_nearest_neighbors(aligned=True)'''
    # TODO: compare nearest neighbors from plot with those from csvs
    # TODO: try with different parameters till it looks good. also with doubles!
    '''visualizations.plot_tsne_according_to_occurrences(words=['Asylbewerber'], k=10, perplexity=5, keep_doubles=False, iter=5000)'''
    # aggregate nearest neighbors of all time in word cloud
    '''experiment.calculate_sum_nearest_neighbors()'''
    '''visualizations.plot_nearest_neighbors_as_word_clouds()'''
    # save distance for each keyword between the aligned epochs
    '''experiment.calculate_cosine_development_for_each_keyword()'''
    # plot these distances
    '''visualizations.plot_cosine_development_each_word()'''
    # plot similarities between certain groups of words over the years
    '''visualizations.plot_exemplary_comparisons()'''
