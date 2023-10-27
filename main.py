import prepare_corpus
import embeddings
import experiment
import visualizations
import utils


if __name__ == '__main__':
    epoch_id = 8
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
    # TODO: alignment first?
    # count frequencies and save in csv
    '''experiment.save_frequency_info_in_csv()'''
    # analyze first occurrences of keywords and save in csv
    '''experiment.create_kw_occurrences_and_merge_to_keyword_list()'''
    # plot frequencies
    '''visualizations.plot_frequencies()
    visualizations.plot_comparing_frequencies()'''
    # save nearest neighbors without alignment
    '''experiment.save_nearest_neighbors()'''
    # possibility to analyse some words more closely, see their context in the corpus
    '''prepare_corpus.print_contexts_for_word_from_lemmatized_corpus('Kugler', 4)'''
    # analyse sentiment of words
    # TODO: maybe omit some less indicative sentiword_models!
    '''experiment.analyse_senti_valuation_of_keywords(sentiword_model="religion")'''
    # plot valuations
    '''visualizations.plot_sentiments(['standard', 'political', 'race'])'''
    # align models to make visualization of nearest neighbors over time
    '''embeddings.align_according_to_occurrences()'''
    # add model for epoch 8 for word "multikulturell"
    # TODO: maybe avoid or include in align_according_to_occurrences?
    '''embeddings.align_two_models(7, 8, 4)'''
    # save nearest neighbors from aligned models to csv
    '''experiment.save_nearest_neighbors(aligned=True)'''
    # TODO: maybe use this csv for the visualization, too! but it doesn't contain any info on the vectors themselves
    '''visualizations.plot_tsne_according_to_occurrences(k=12, keep_doubles=False)'''
    # save distance for each keyword between the aligned epochs
    '''experiment.compare_connotations_for_all_keywords()'''
    # plot these distances
    visualizations.plot_comparing_connotations()
