import os.path

import utils
from collections import Counter
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np


# Frequency
def save_frequency_info_in_csv():
    output_file_path = 'results/freqs.csv'
    utils.write_info_to_csv(output_file_path, ['epoch', 'keyword', 'count', 'freq', 'pMW'])
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
            # calculate frequency per million words
            pMW = freq * 1000000
            utils.write_info_to_csv(output_file_path, [epoch, kw, count, freq, pMW], 'a')


def prepared_corpus_to_wordlist(corpus_name):
    corpus_unflattened = utils.unpickle(corpus_name)
    # flatten corpus to one layered list
    return [word for sent in corpus_unflattened for word in sent]


def find_counts_for_keywords(count_dict):
    keywords = utils.load_keywords()
    return {k: count_dict.get(k, 0) for k in keywords}


def include_written_form_in_expected_csv(method):
    df = pd.read_csv('data/expected_values.csv')
    translation_df = pd.read_csv(f'data/expected_{method}_translation.csv')
    expected_values = df[f'expected_{method}'].tolist()
    expected_written = []
    for val in expected_values:
        written = translation_df[translation_df[f'{method}_value'] == val]['written'].iloc[0]
        expected_written.append(written)
    df[f'written_{method}'] = expected_written
    df.to_csv('data/expected_values.csv', index=False)


def make_freq_slices():
    freqs_df = pd.read_csv('results/freqs.csv')
    all_freqs = freqs_df['pMW'].tolist()
    all_freqs_arr = np.array(all_freqs)
    sorted_freqs = sorted(all_freqs_arr[all_freqs_arr != 0.0])
    # create 8 slices
    slices = np.array_split(sorted_freqs, 8)
    # guarantee smooth transitions without noticeable gaps
    for i in range(len(slices) - 1):
        slices[i][-1] = slices[i + 1][0] - 0.0001
    data = {'expected_freq_key': range(1, 9),
            'pMW_mean': [np.mean(s) for s in slices],
            'pMW_max': [max(s) for s in slices],
            'pMW_min': [min(s) for s in slices]}
    df = pd.DataFrame(data)
    df.to_csv('data/results/expected_freq_results_slices.csv', index=False)


def find_first_occurrences_for_keywords():
    # load keywords
    df = pd.read_csv('data/keywords.csv')
    keywords = df['keyword'].tolist()
    # load freqs
    df = pd.read_csv('results/freqs.csv')
    output_file_path = "results/kw_occurrences.csv"
    utils.write_info_to_csv(output_file_path, ['keyword', 'ignore', 'first_occ_epoch', 'last_occ_epoch',
                                                                'loophole'])
    # for each keyword find first & last non-null freq-epoch
    for kw in keywords:
        kw_freqs_df = df[df['keyword'] == kw]
        # freq infos for kw df
        enough_occ_epochs_df = kw_freqs_df[kw_freqs_df['count'] > 4]
        # for which epoch is count > 4? Only these will generate results for the word embedding model
        enough_occ_epochs = enough_occ_epochs_df['epoch'].tolist()
        first_occ_epoch = min(enough_occ_epochs) if enough_occ_epochs else 0
        last_occ_epoch = max(enough_occ_epochs) if enough_occ_epochs else 0
        loopholes = []
        for i in range(first_occ_epoch, last_occ_epoch+1):
            if i not in enough_occ_epochs:
                loopholes.append(f'{i}')
        # ignore keyword if it does not occur at all
        ignore = 1 if first_occ_epoch == last_occ_epoch == 0 and len(loopholes) == 1 and int(loopholes[0]) == 0 else 0
        # write into csv: keyword, ignore, first_occ_epoch, last_occ_epoch, loopholes
        utils.write_info_to_csv(output_file_path, [kw, ignore, first_occ_epoch, last_occ_epoch,
                                                   '_'.join(loopholes) if len(loopholes) > 0 else 0], mode='a')


def create_kw_occurrences_and_merge_to_keyword_list():
    find_first_occurrences_for_keywords()
    df1 = pd.read_csv('data/keywords.csv')
    df2 = pd.read_csv('results/kw_occurrences.csv')
    merged_df = df1.merge(df2, on='keyword', how='outer')
    merged_df.to_csv('data/keywords_merged.csv', index=False)


def calculate_mean_frequency_over_all_epochs():
    # iterate freqs.csv
    df = pd.read_csv('results/freqs.csv')
    keywords = list(set(df['keyword'].tolist()))
    # for each kw, get pMW and mean
    results = [{'keyword': kw, 'mean_freq': np.mean(np.array(df[df['keyword'] == kw]['pMW'].tolist())) } for kw in keywords]
    # Get indices that would sort mean_freqs
    sorted_indices = np.argsort([results[i]['mean_freq'] for i in range(len(results))])[::-1]
    # Create a new array with ranks
    for i, idx in enumerate(sorted_indices):
        results[idx]['rank'] = i + 1
    return sorted(results, key=lambda x: x['rank'], reverse=True)


def get_freq_maxima_for_epochs():
    df = pd.read_csv('results/freqs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    keywords = keywords_df['keyword'].tolist()
    indices = []
    # iterate keywords
    for kw in keywords:
        kw_df = df[df['keyword'] == kw]
        # if not all pMW values are the same (otherwise, finding the maximum would not be meaningful)
        if kw_df['pMW'].nunique() > 1:
            # find index of maximum freq (per million words)
            index_of_max_freq = kw_df['pMW'].idxmax()
            indices.append(index_of_max_freq)
    # make df with only rows containing a minimum value of a keyword
    reduced_df = df.iloc[indices]
    # filter df for combined words: only keep the smaller value, if a word is combined with another one
    for combined_index in [index for index in set(keywords_df['combine_group']) if index != 0]:
        combined_words = keywords_df[keywords_df['combine_group'] == combined_index]['keyword'].tolist()
        combined_df = reduced_df[reduced_df['keyword'].isin(combined_words)]
        # drop smaller value if there are two
        if len(combined_df) > 1:
            bigger_idx = combined_df['pMW'].idxmin()
            reduced_df = reduced_df.drop([bigger_idx])
    # return maxima count
    max_count_dict = reduced_df.groupby('epoch').size().to_dict()
    for epoch_id in range(1, 9):
        epoch_name = utils.get_epoch_written_form_short(epoch_id)
        max_count_dict[epoch_id] = {'count': max_count_dict.get(epoch_id, 0), 'name': epoch_name}
    return dict(sorted(max_count_dict.items()))


# Sentiment
def analyse_senti_valuation_of_keywords(sentiword_set=""):
    print("preparing sentiment analysis")
    # load keywords
    keyword_df = pd.read_csv('data/keywords_merged.csv')
    keywords = keyword_df['keyword'].tolist()
    # load sentiwords according to choice
    if sentiword_set == 'combination':
        senti_file_path1 = f"data/sentiwords.csv"
        senti_file_path2 = f"data/political_sentiwords.csv"
        df1 = pd.read_csv(senti_file_path1)
        df2 = pd.read_csv(senti_file_path2)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        senti_file_path = f"data/{sentiword_set}{'_' if sentiword_set else ''}sentiwords.csv"
        df = pd.read_csv(senti_file_path)
    # group sentiwords by value (A: +1/B: -1)
    df_pos = df[df["value"] == 1]
    df_neg = df[df["value"] == -1]
    pos_words = df_pos["word"].tolist()
    neg_words = df_neg["word"].tolist()
    # prepare output csv
    output_file_path = "results/senti.csv"
    if not os.path.exists(output_file_path):
        utils.write_info_to_csv(output_file_path, ["word", "epoch", "sentiword_set", "value"])
    # iterate keywords
    for kw in keywords:
        kw_keyword_df = keyword_df[keyword_df['keyword'] == kw]
        if kw_keyword_df['ignore'].iloc[0] == 0:
            print(f"Analysing key word {kw}")
            # for keyword: iterate epochs
            for epoch in range(1, 9):
                # get associated wordvectors
                word_vectors = KeyedVectors.load(f"data/models/base_models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors")
                # only analyze word if it is supposed to be in the vocab
                if epoch in range(kw_keyword_df['first_occ_epoch'].iloc[0], kw_keyword_df['last_occ_epoch'].iloc[0] + 1) and str(epoch) not in kw_keyword_df['loophole'].iloc[0]:
                    # calculate bias value of word with WEAT method
                    senti = weat(kw, pos_words, neg_words, word_vectors)
                    # save value in csv
                    utils.write_info_to_csv(output_file_path,
                                            [kw, epoch, sentiword_set if sentiword_set else "standard", senti],
                                            mode="a")


# adapted from https://github.com/leahannah/weat_demo/blob/main/weat.py
def weat(w, A, B, word_vectors):
    """Calculate bias score of attribute word w and two target sets A and B
    :param w: attribute word
    :param A: target set
    :param B: other target set
    :param word_vectors: keyedvectors to use"""
    sim_wa = [word_vectors.similarity(w, a) for a in A if a in word_vectors.index_to_key]
    sim_wb = [word_vectors.similarity(w, b) for b in B if b in word_vectors.index_to_key]
    return np.mean(sim_wa) - np.mean(sim_wb)


def make_senti_slices(sentiword_sets=['standard', 'political', 'combination']):
    senti_df = pd.read_csv('results/senti.csv')
    expected_senti_keys = []
    senti_mean = []
    senti_max = []
    senti_min = []
    sentiword_set_list = []
    for sentiword_set in sentiword_sets:
        all_senti = senti_df[senti_df['sentiword_set'] == sentiword_set]['value'].tolist()
        # split at 0
        negative_values = sorted([val for val in all_senti if val < 0])
        positive_values = sorted([val for val in all_senti if val > 0])
        # take set with the higher range
        if max([abs(val) for val in negative_values]) > abs(max(positive_values)):
            first_list = negative_values
        else:
            first_list = positive_values
        # create 9 equally big slices for the first list
        first_slices = np.array_split(first_list, 9)
        first_slices_combined = [[] for _ in range(4)]
        # combine first 8 arrays to 4 big ones, add smaller slice to the end
        for i in [0, 2, 4, 6]:
            first_slices_combined[int(i * 0.5)] = np.concatenate((first_slices[i], first_slices[i + 1]))
        neutral_slice_first_half = first_slices[8]
        # guarantee smooth transitions without noticeable gaps
        for i in range(len(first_slices_combined) - 1):
            first_slices_combined[i][-1] = first_slices_combined[i + 1][0] - 0.0001
        # create 4 slices for the second set, but according to the ranges of the first set to obtain symmetry
        second_slices = [[0 - val for val in nested_list[::-1]] for nested_list in first_slices_combined]
        second_slices.reverse()
        slices = []
        # create slice for neutral values, ensuring no gaps
        # and unite all slices to one nested list in ascending order
        if first_slices_combined[0][0] < 0:
            neutral_slice = [[neutral_slice_first_half[0] - 0.0001, 0.0, -neutral_slice_first_half[0] + 0.0001]]
            slices = first_slices_combined + neutral_slice + second_slices
        elif second_slices[0][0] < 0:
            neutral_slice = [[-neutral_slice_first_half[0] - 0.0001, 0.0, neutral_slice_first_half[0] + 0.0001]]
            slices = second_slices + neutral_slice + first_slices_combined
        expected_translation_df = pd.read_csv('data/expected_senti_translation.csv')
        expected_senti_keys.extend(sorted(expected_translation_df['senti_value'].tolist()[:9]))
        senti_mean.extend([np.mean(s) for s in slices])
        senti_max.extend([max(s) for s in slices])
        senti_min.extend([min(s) for s in slices])
        sentiword_set_list.extend([sentiword_set]*9)
    data = {'expected_senti_key': expected_senti_keys,
                'senti_mean': senti_mean,
                'senti_max': senti_max,
                'senti_min': senti_min,
                'sentiword_set': sentiword_set_list}
    df = pd.DataFrame(data)
    df.to_csv('data/results/expected_senti_results_slices.csv', index=False)


def calculate_mean_sentiment_over_all_epochs(sentiword_set="combination"):
    # iterate senti.csv
    df = pd.read_csv('results/senti.csv')
    keywords = list(set(df['word'].tolist()))
    # for each kw, get senti and mean
    results = [{'word': kw, 'mean_senti': np.mean(np.array(df[(df['word'] == kw) & (df['sentiword_set'] == sentiword_set)]
                                                           ['value'].tolist()))} for kw in keywords]
    # Get indices that would sort mean_senti
    sorted_indices = np.argsort([results[i]['mean_senti'] for i in range(len(results))])[::-1]
    # Create a new array with ranks
    for i, idx in enumerate(sorted_indices):
        results[idx]['rank'] = i + 1
    return sorted(results, key=lambda x: x['rank'], reverse=True)


def get_senti_minima_for_epochs(sentiword_set='combination'):
    df = pd.read_csv('results/senti.csv')
    filtered_df = df[df['sentiword_set'] == sentiword_set]
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    keywords = keywords_df['keyword'].tolist()
    indices = []
    # iterate keywords
    for kw in keywords:
        kw_df = filtered_df[filtered_df['word'] == kw]
        # if not all values are the same (then finding the minimum would not make sense)
        if kw_df['value'].nunique() > 1:
            # find index and save of minimum value
            index_of_min_senti = kw_df['value'].idxmin()
            indices.append(index_of_min_senti)
    # make df with only rows containing a minimum value of a keyword
    reduced_df = df.iloc[indices]
    # filter df for combined words: only keep the smaller value, if a word is combined with another one
    for combined_index in [index for index in set(keywords_df['combine_group']) if index != 0]:
        combined_words = keywords_df[keywords_df['combine_group'] == combined_index]['keyword'].tolist()
        combined_df = reduced_df[reduced_df['word'].isin(combined_words)]
        # drop bigger value if there are two
        if len(combined_df) > 1:
            bigger_idx = combined_df['value'].idxmax()
            reduced_df = reduced_df.drop([bigger_idx])
    # return minima count
    min_count_dict = reduced_df.groupby('epoch').size().to_dict()
    for epoch_id in range(1, 9):
        epoch_name = utils.get_epoch_written_form_short(epoch_id)
        min_count_dict[epoch_id] = {'count': min_count_dict[epoch_id], 'name': epoch_name}
    return min_count_dict


# Word Associations
def save_nearest_neighbors(aligned=False):
    if aligned:
        df = pd.read_csv('data/keywords_merged.csv')
    epochs = range(1, 9)
    base_folder = 'data/models/base_models'
    # load keywords from csv
    keywords = utils.load_keywords()
    # initialize csv to save results to
    words_similarities_headings = []
    for no in range(1, 11):
        words_similarities_headings.append(f'Word_{no}')
        words_similarities_headings.append(f'Similarity_{no}')
    save_file_path = f'results/nearest_neighbors{"_aligned" if aligned else ""}.csv'
    utils.write_info_to_csv(save_file_path, ['Keyword', 'Epoch'] + words_similarities_headings)
    # iterate keywords
    for kw in keywords:
        if aligned:
            row = df[df['keyword'] == kw].iloc[0]
            base_folder = f'data/models/aligned_models/start_epoch_{row.first_occ_epoch}{f"_lh_{row.loophole}" if not str(0) in row.loophole else ""}'
            epochs = [item for item in range(row.first_occ_epoch, row.last_occ_epoch + 1) if
                                str(item) not in row.loophole]
        # iterate base models/epochs
        for epoch in epochs:
            word_vectors_path = f'{base_folder}/epoch{epoch}_lemma_200d_7w_cbow{"_aligned" if aligned else ""}.wordvectors'
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


def prepare_target_vectors_for_tsne(epochs, target_word, aligned_base_folder, k=15, keep_doubles=False):
    target_vectors = {}
    for epoch in epochs:
        epoch_written = utils.get_epoch_written_form_short(epoch)
        model_wv_path = f'{aligned_base_folder}/epoch{epoch}_lemma_200d_7w_cbow{"_aligned" if len(epochs) > 1 else ""}.wordvectors'
        # print("Year: ", model_years[year_index])
        word_vectors = KeyedVectors.load(model_wv_path)
        vocab = list(word_vectors.index_to_key)
        target_word_year = f'{target_word}_{epoch_written}'
        target_vectors[target_word_year] = {}
        target_vectors[target_word_year]['vector'] = word_vectors[target_word]
        target_vectors[target_word_year]['type'] = 'target_word'
        word_sim = word_vectors.most_similar(positive=target_word, topn=k)
        for ws in word_sim:
            if keep_doubles:
                ws_key = f'{ws[0]}_{epoch_written}'
                target_vectors[ws_key] = {}
                target_vectors[ws_key]['vector'] = word_vectors[ws[0]]
                target_vectors[ws_key]['type'] = epoch_written
                target_vectors[ws_key]['sim'] = ws[1]
            elif (ws[0] not in target_vectors.keys()) or (ws[0] in target_vectors.keys()
                                                          and ws[1] > target_vectors[ws[0]]['sim']):
                # throwing away double existing words makes them invisible in future epochs
                target_vectors[ws[0]] = {}
                target_vectors[ws[0]]['vector'] = word_vectors[ws[0]]
                target_vectors[ws[0]]['type'] = epoch_written
                target_vectors[ws[0]]['sim'] = ws[1]
    return target_vectors


def calculate_sum_nearest_neighbors(aligned=False):
    # prepare output
    out_path = 'results/aggregated_nearest_neighbors.csv'
    words_similarities_headings = []
    for no in range(1, 21):
        words_similarities_headings.append(f'Word_{no}')
        words_similarities_headings.append(f'Similarity_{no}')
    utils.write_info_to_csv(out_path, ['Keyword'] + words_similarities_headings)
    # read data
    df = pd.read_csv(f'results/nearest_neighbors{"_aligned" if aligned else ""}.csv')
    # for each keyword:
    keywords = set(df['Keyword'].tolist())
    for kw in keywords:
        # get all words word_1 to word_10 with their resp. similarities
        neighbors = []
        kw_df = df[df['Keyword'] == kw]
        for index, row in kw_df.iterrows():
            for i in range(1, 11):
                word = row[f'Word_{i}']
                sim = row[f'Similarity_{i}']
                existing_object = next((obj for obj in neighbors if obj['word'] == word), None)
                # if they appear more than once, sum similarity
                if existing_object:
                    existing_object['similarity'] = sum([existing_object['similarity'], sim])
                else:
                    new_word_object = {'word': word, 'similarity': sim}
                    neighbors.append(new_word_object)
        # sort the aggregated neighbors by similarity
        sorted_neighbors = sorted(neighbors, key=lambda d: d['similarity'], reverse=True)
        output_list = [item for sublist in ([word['word'], word['similarity']] for word in sorted_neighbors[:20]) for item in sublist]
        # for each key word save the 20 nearest neighbors of all epochs with resp. similarities
        utils.write_info_to_csv(out_path, [kw] + output_list, 'a')


def calculate_cosine_similarity_between_word_group(main_word, other_words, necessary_epochs):
    # initialize result variable
    results = []
    # doing this, also take care of cases where one of the words does not exist in epoch
    for w in other_words:
        results.append([])
        for epoch in necessary_epochs:
            word_vectors = KeyedVectors.load(f'data/models/base_models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors')
            try:
                val = word_vectors.similarity(main_word, w)
            except KeyError:
                print(f"One of the words {main_word}, {w} does not exist in epoch {epoch}!")
                val = None
            results[other_words.index(w)].append(val)
            # TODO: avoid that one list is only [None]?
    return results


def print_contexts_for_word_from_lemmatized_corpus(word, epoch):
    """
    print all sentences in which a specific word appears in an epoch. This method can help understanding
    the corpus better, find lemmatizing mistakes and explain possible results of the experiment
    :param word: word to look for
    :param epoch: number signifying an historical epoch defined in epochs.csv
    :return: all sentences that contain the word (tokenized, lemmatized)
    """
    # find path to corpus
    corpus_path = f'data/corpus/epoch{epoch}_prepared_lemma'
    # unpickle corpus
    corpus = utils.unpickle(corpus_path)
    sents_containing_word = []
    # search lists for word
    for sent in corpus:
        if word in sent:
            sents_containing_word.append(sent)
    print(f'{len(sents_containing_word)} Sätze mit Wort {word} gefunden:')
    for s in sents_containing_word:
        print(f'{sents_containing_word.index(s)}. {s}\n')
    return sents_containing_word
