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
    # plot frequencies
    '''for absolute in [True, False]:
        visualizations.plot_frequencies(absolute=absolute)
    visualizations.plot_comparing_frequencies()'''
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
    # evaluate models
    '''for epoch_id in range(1, 9):
        word_vectors = KeyedVectors.load(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.wordvectors')
        print(f'epoch: {epoch_id}, vocabsize: {len(word_vectors)}')'''
    # analyse sentiment of words
    '''for sentiword_set in ['', 'political', 'combination']:
        experiment.analyse_senti_valuation_of_keywords(sentiword_set=sentiword_set)'''
    # plot valuations
    '''visualizations.plot_sentiments(['combination'])
    visualizations.plot_comparing_sentiments()'''
    # save nearest neighbors without alignment
    '''experiment.save_nearest_neighbors()'''
    # possibility to analyse some words more closely, see their context in the corpus
    '''prepare_corpus.print_contexts_for_word_from_lemmatized_corpus('motzen', 1)'''
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
    visualizations.plot_tsne_according_to_occurrences(words=['Ausländer'], k=10, perplexity=10, keep_doubles=False, iter=5000)
    # save distance for each keyword between the aligned epochs
    '''experiment.calculate_cosine_development_for_each_keyword()'''
    # plot these distances
    '''visualizations.plot_cosine_development_each_word()'''
    # plot similarities between certain groups of words over the years
    '''visualizations.plot_exemplary_comparisons()'''
    # upgrade expected_values.csv with translation for senti
    # TODO: include earlier in process, same for freqs, maybe make own method/function
    '''df = pd.read_csv('data/expected_values.csv')
    translation_df = pd.read_csv('data/expected_senti_translation.csv')
    expected_senti = df['expected_valuation'].tolist()
    expected_senti_written = []
    for f in expected_senti:
        written = translation_df[translation_df['senti_value'] == f]['written'].iloc[0]
        expected_senti_written.append(written)
    df['written_senti'] = expected_senti_written
    df.to_csv('data/expected_values.csv')'''
