import prepare_corpus
import embeddings
import experiment
import visualizations
import utils
from gensim.models import KeyedVectors


if __name__ == '__main__':
    # Corpus preprocessing
    '''epoch_id = 1'''
    # unite text for epochs. This process might take a while
    '''prepare_corpus.pure_text_to_epoch_txt(epoch_id)'''
    # preprocess text and save this step. This process might take a while
    '''data = prepare_corpus.prepare_text_for_embedding_training(f"data/corpus/epoch{epoch_id}.txt", lemmatize=True)
    utils.make_pickle(f"data/corpus/epoch{epoch_id}_prepared_lemma", data)'''

    # Frequency analysis (prerequisite: corpus preparation)
    # count frequencies and save in csv
    '''experiment.analyse_frequency_of_keywords()'''
    # analyze first occurrences of keywords and save in keywords_merged.csv
    '''experiment.create_kw_occurrences_and_merge_to_keyword_list()'''
    # append expected_values.csv with written form to enable plotting
    '''experiment.include_written_form_in_expected_csv(method='freq')'''
    # save pMW slices of freqs results to enable comparing with "freq classes"
    '''experiment.make_freq_slices()'''
    # plot frequencies
    '''for relative in [True, False]:
        visualizations.plot_frequencies(relative=relative)'''
    '''visualizations.plot_comparing_frequencies()'''
    # calculate and plot mean frequencies
    '''visualizations.plot_mean_frequencies_for_keywords_as_bar_plot()'''
    '''visualizations.plot_frequency_maxima_for_epochs_as_bar_plot()'''

    # Embedding training and alignment (prerequisite: frequency analysis)
    # use preprocessed text to train and save cbow models
    '''for epoch_id in range(1, 9):
        prepared_corpus = utils.unpickle(f"data/corpus/epoch{epoch_id}_prepared_lemma")
        model = embeddings.make_word_emb_model(prepared_corpus)
        # save model
        print("Saving model")
        model.save(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.model')
        # save word vectors
        word_vectors = model.wv
        word_vectors.save(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.wordvectors')'''
    # optional: evaluate the models' vocab sizes
    '''for epoch_id in range(1, 9):
        word_vectors = KeyedVectors.load(f'data/models/base_models/epoch{epoch_id}_lemma_200d_7w_cbow.wordvectors')
        print(f'epoch: {epoch_id}, vocabsize: {len(word_vectors)}')'''
    # align models (necessary to make tsne visualization of nearest neighbors over time)
    '''embeddings.align_according_to_occurrences()'''
    # add some missing aligned models, following the warnings from process before.
    #  manually checked if adding these would be valid for the resp. folders
    '''embeddings.align_two_models(5, 6, 4)
    embeddings.align_two_models(6, 7, 4)
    embeddings.align_two_models(7, 8, 4)
    embeddings.align_two_models(7, 8, 6)'''

    # Sentiment analysis with WEAT (prerequisite: embedding training)
    # analyse sentiment of words
    '''for sentiword_set in ['', 'political', 'combination']:
        experiment.analyse_senti_valuation_of_keywords(sentiword_set=sentiword_set)'''
    # append expected.csv with written form to enable plotting
    '''experiment.include_written_form_in_expected_csv(method='senti')'''
    # save senti slices of senti results for plots
    '''experiment.make_senti_slices()'''
    # plot senti
    '''visualizations.plot_sentiments(['combination'], include_expected=True, show_result_groups=True)
    visualizations.plot_comparing_sentiments()'''
    # calculate and plot mean sentiment over all time
    '''visualizations.plot_mean_sentiments_for_keywords_as_bar_plot()'''
    '''visualizations.plot_senti_minima_for_epochs_as_bar_plot()'''

    # Word association analysis (prerequisite: embedding training and alignment)
    # save nearest neighbors with and without alignment
    '''experiment.analyse_word_associations()'''
    '''experiment.analyse_word_associations(aligned=True)'''
    # save aggregated nearest neighbors for each keyword
    '''experiment.calculate_sum_nearest_neighbors_per_keyword()'''
    # possibility to analyse some words more closely, see their context in the corpus (of a specific epoch)
    '''experiment.print_contexts_for_word_from_lemmatized_corpus('Vertriebene', 1)'''
    # plot aligned nearest neighbors as tsne plot
    '''visualizations.plot_tsne_according_to_occurrences(words=['Asylant'], k=8, perplexity=5, keep_doubles=False, 
                                                      iterations=1000)'''
    # plot cosine similarities between certain groups of words over the years
    '''visualizations.plot_cosine_developments_of_word_groups()'''
    # plot development of aggregated nearest neighbors with heatmap
    '''visualizations.plot_nearest_neighbors_heatmap()'''
