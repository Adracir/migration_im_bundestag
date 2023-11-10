import prepare_corpus
import embeddings
import experiment
import visualizations
import utils
from gensim.models import KeyedVectors
from collections import Counter
import math


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
    # get most frequent word & its occurrence for each epoch
    '''for epoch in range(1, 9):
        wv = KeyedVectors.load(f'data/models/base_models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors')
        most_frequent_word = wv.index_to_key[0]
        wordlist = experiment.prepared_corpus_to_wordlist(f"data/corpus/epoch{epoch}_prepared_lemma")
        dict = Counter(wordlist)
        print(f"Epoch {epoch}: most frequent word '{most_frequent_word}' appears {dict.get(most_frequent_word, 0)} times!")'''
    # add frequency classes to freqs csv
    '''experiment.determine_frequency_classes_for_keywords()'''
    # print number of freq classes for each epoch
    '''highest_freq_for_epochs = [2874025, 1927682, 2229283, 3214476, 3232228, 4930583, 2840717, 987578]
    for epoch in range(1, 9):
        print(f"Epoch {epoch}: {round(math.log2(highest_freq_for_epochs[epoch - 1] / 1))}")
    max_freq_class = [21, 21, 21, 22, 22, 22, 21, 20]'''
    # analyze first occurrences of keywords and save in keywords csv
    '''experiment.create_kw_occurrences_and_merge_to_keyword_list()'''
    # plot corpus frequency distribution
    '''visualizations.plot_frequency_distribution_for_corpora()'''
    # plot frequencies
    visualizations.plot_frequencies(include_expected=True, absolute=True, logarithmic=True)
    # visualizations.plot_frequencies()
    '''visualizations.plot_comparing_frequencies()'''
    # plot frequency classes
    '''visualizations.plot_frequency_classes(include_expected=True, absolute=True)'''
    # save nearest neighbors without alignment
    '''experiment.save_nearest_neighbors()'''
    # possibility to analyse some words more closely, see their context in the corpus
    '''prepare_corpus.print_contexts_for_word_from_lemmatized_corpus('Kugler', 4)'''
    # evaluate sentiment models
    '''experiment.evaluate_senti_words_stability(['', 'race', 'religion', 'political', 'ethics'])'''
    # analyse sentiment of words
    '''experiment.analyse_senti_valuation_of_keywords(sentiword_set="religion")'''
    '''for senti_word_set in ['', 'religion', 'political', 'race', 'ethics']:
        experiment.analyse_senti_valuation_of_keywords(sentiword_set=senti_word_set, with_axis=True)'''
    # plot valuations
    '''visualizations.plot_sentiments(['standard', 'political'], with_axis=True, include_expected=False)'''
    # align models to make visualization of nearest neighbors over time
    '''embeddings.align_according_to_occurrences()'''
    # add model for epoch 8 for word "multikulturell"
    # TODO: maybe avoid or include in align_according_to_occurrences?
    '''embeddings.align_two_models(7, 8, 4)'''
    # save nearest neighbors from aligned models to csv
    '''experiment.save_nearest_neighbors(aligned=True)'''
    # TODO: compare nearest neighbors from plot with those from csvs
    # TODO: try with different parameters till it looks good. also with doubles
    '''visualizations.plot_tsne_according_to_occurrences(k=12, perplexity=5, keep_doubles=False, iter=5000)'''
    # save distance for each keyword between the aligned epochs
    '''experiment.compare_connotations_for_all_keywords()'''
    # plot these distances
    '''visualizations.plot_comparing_connotations(horizontal=True, smooth=True)'''
    # plot similarities between certain groups of words over the years
    '''visualizations.plot_exemplary_comparisons()'''
