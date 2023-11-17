import experiment
import utils
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE  # needed to be installed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # needed to be installed
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import collections as mc
import os
from scipy.interpolate import make_interp_spline

from scipy.integrate import quad


# Helper methods
def ignore_1000_vals(x, y):
    x_segments = []
    y_segments = []
    current_segment_x = []
    current_segment_y = []
    for xi, yi in zip(x, y):
        if yi != 1000:
            current_segment_x.append(xi)
            current_segment_y.append(yi)
        else:
            if current_segment_x:
                x_segments.append(current_segment_x)
                y_segments.append(current_segment_y)
            current_segment_x = []
            current_segment_y = []
    if current_segment_x:
        x_segments.append(current_segment_x)
        y_segments.append(current_segment_y)
    return [x_segments, y_segments]


def transform_expected_freqs_values(all_freqs, relevant_expected_values):
    transformed_values = []
    max_freqs = max(all_freqs)
    step_size = max_freqs/8
    for val in relevant_expected_values:
        if val == 1000:
            transformed_values.append(1000)
        else:
            transformed_values.append(step_size * val)
    return transformed_values


def transform_expected_senti_values(all_senti, expected_values):
    transformed_values = []
    max_senti = max(all_senti, key=abs)
    for val in expected_values:
        if val == 1000:
            transformed_values.append(1000)
        else:
            transformed_values.append(val * abs(max_senti))
    return transformed_values


# Frequency methods
def plot_frequencies(include_expected=True, absolute=False):
    freqs_df = pd.read_csv('data/results/freqs.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    if include_expected:
        expected_df = pd.read_csv('data/expected_values.csv')
    # keywords = utils.load_keywords()
    keywords = keywords_df['keyword'].tolist()
    combine_groups = keywords_df['combine_group'].tolist()
    ignoring = keywords_df['ignore'].tolist()
    epochs = epochs_df['epoch_id'].tolist()
    # TODO: Obacht! freqs and keywords and epochs should be at same status for this to work!
    #  Maybe implement cleverer way
    indices_done = []
    for i in range(0, len(keywords)):
        combine_group = combine_groups[i]
        # case 1: word has already been plotted
        if i in indices_done or ignoring[i]:
            continue
        # case 2: word does not need to be combined with other spelling and has not been plotted yet
        if not combine_group:
            kw = keywords[i]
            kw_freqs_df = freqs_df[freqs_df['keyword'] == kw]
            freqs = kw_freqs_df['freq'].tolist()
            if include_expected:
                reference_values = freqs if not absolute else freqs_df['freq'].tolist()
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(kw)]
                expected_values = expected_kw_df['expected_freq'].tolist()
                # transform values using all existing freqs values
                transformed_exp_values = transform_expected_freqs_values(reference_values, expected_values)
            title = f'Häufigkeiten des Schlagwortes {kw}'
            path = f'data/results/plots/frequencies/freq_{kw}{"_with_expected" if include_expected else ""}{"_abs_ref" if absolute else ("_rel_ref" if include_expected else "")}{"_log" if logarithmic else ""}_plot.png'
        # case 3: combine different spellings
        else:
            # find all spellings for word
            combine_group_indices = [i for i, x in enumerate(combine_groups) if x == combine_group]
            freqs = [0] * len(epochs)
            combined_kws = []
            for cgi in combine_group_indices:
                kw = keywords[cgi]
                combined_kws.append(kw)
                kw_freqs_df = freqs_df[freqs_df['keyword'] == kw]
                kw_freqs = kw_freqs_df['freq'].tolist()
                # save sum of all kw_freqs into freqs
                freqs = [sum(x) for x in zip(freqs, kw_freqs)]
            if include_expected:
                reference_values = freqs_df['freq'].tolist() if absolute else freqs
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(keywords[combine_group_indices[0]])]
                expected_values = expected_kw_df['expected_freq'].tolist()
                # transform values using all existing freqs values
                transformed_exp_values = transform_expected_freqs_values(reference_values, expected_values)
            indices_done.extend(combine_group_indices)
            title = f"Häufigkeiten der Schlagwörter {', '.join(combined_kws)}"
            path = f"data/results/plots/frequencies/freq_combined_{'_'.join(combined_kws)}{'_with_expected' if include_expected else ''}" \
                   f"{'_abs_ref' if absolute else ('_rel_ref' if include_expected else '')}{'_log' if logarithmic else ''}_plot.png"
        written_forms = epochs_df['written_form'].tolist()  # TODO: assumes same order of epochs, maybe improve
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('relative Häufigkeiten im Korpus')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.grid(True)
        # plot a grey area for the expected values
        if include_expected:
            without_1000_vals = ignore_1000_vals(x, transformed_exp_values)
            x_segments = without_1000_vals[0]
            y_segments = without_1000_vals[1]
            plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', alpha=0.5, linewidth=70)
            plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', linewidth=1,
                 label='expected values with error band')
        # plot freqs
        plt.plot(x, freqs, 'r-', label="frequencies")
        # plot legend
        plt.legend()
        # set tight layout (so that nothing is cut out)
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {i} saved")


def plot_frequency_distribution_for_corpora():
    for epoch in range(1, 9):
        dict = experiment.total_word_frequency_distribution(epoch)
        sorted_freqs = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        words, freqs = zip(*sorted_freqs)
        ranks = list(range(1, len(words) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(ranks[:len(ranks)//500], freqs[:len(ranks)//500], linestyle='-', color='b', alpha=0.5)
        plt.suptitle(f'Verteilung der Worthäufigkeit in Relation zum Rang, Epoche {epoch}', fontsize=16, fontweight="bold")
        plt.title(f'Gesamtgröße des Vokabulars: {len(ranks)} Wörter. X-Achse zur Übersichtlichkeit begrenzt')
        plt.xlabel('Rang')
        plt.ylabel('Häufigkeit')
        plt.grid(True)
        plt.xlim(0, len(ranks)//500)
        fig = plt.gcf()
        fig.savefig(f'data/results/plots/frequencies/distribution_epoch{epoch}.png')
        plt.close(fig)
        print(f'epoch {epoch} plotted')


def plot_comparing_frequencies():
    freqs_df = pd.read_csv('data/results/freqs.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    # keywords = utils.load_keywords()
    keywords = keywords_df['keyword'].tolist()
    compare_groups = keywords_df['freq_compare_group'].tolist()
    epochs = epochs_df['epoch_id'].tolist()
    # TODO: Obacht! freqs and keywords and epochs should be at same status for this to work!
    #  Maybe implement cleverer way
    indices_done = []
    for i in range(0, len(keywords)):
        compare_group = compare_groups[i]
        # case 1: word has already been plotted or no compare group is available for this word
        if i in indices_done or compare_groups[i] == 0:
            continue
        # case 2: word should be plotted in comparison with other words
        # find all words that should be plotted
        compare_group_indices = [i for i, x in enumerate(compare_groups) if x == compare_group]
        freqs = [0] * len(compare_group_indices)
        compared_kws = []
        for c in range(0, len(compare_group_indices)):
            kw = keywords[compare_group_indices[c]]
            compared_kws.append(kw)
            kw_freqs_df = freqs_df[freqs_df['keyword'] == kw]
            kw_freqs = kw_freqs_df['freq'].tolist()
            # save freqs of kw in corresponding list in list
            freqs[c] = kw_freqs
        indices_done.extend(compare_group_indices)
        title = f"Häufigkeiten der Schlagwörter {', '.join(compared_kws)}"
        path = f"data/results/plots/frequencies/freq_compared_{'_'.join(compared_kws)}_plot.png"
        written_forms = epochs_df['written_form'].tolist()  # TODO: assumes same order of epochs, maybe improve
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('relative Häufigkeiten im Korpus')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.grid(True)
        # plot freqs
        colors = ['r', 'b', 'g', 'y']
        for a in range(0, len(freqs)):
            plt.plot(x, freqs[a], f'{colors[a]}-', label=compared_kws[a])
        # show legend
        plt.legend()
        # set tight layout (so that nothing is cut out)
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {i} saved")


# TODO: möglich, auch für Wertungen und Konnotationen comparing und combining der Wörter zu betreiben? Evtl alle
#  Informationen zu Kombinationen etc. sprechend in keywords_merged.csv speichern!
# Sentiments/Valuation
# TODO: Idee: Expected Values immer absolut einzeichnen, und zusätzlich noch den Plot ohne with_expected speichern.
def plot_sentiments(sentiword_set_arr, with_axis=False, include_expected=False, absolute=False):
    senti_df = pd.read_csv(f'data/results/senti{"_with_axis" if with_axis else ""}.csv')
    senti_df_filtered = senti_df[senti_df['sentiword_set'].isin(sentiword_set_arr)]
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    if include_expected:
        expected_df = pd.read_csv('data/expected_values.csv')
    # keywords = utils.load_keywords()
    keywords = keywords_df['keyword'].tolist()
    ignoring = keywords_df['ignore'].tolist()
    for kw in keywords:
        if ignoring[keywords.index(kw)] != 1:
            kw_senti_df = senti_df_filtered[senti_df_filtered['word'] == kw]
            # only take epochs that are valid
            epochs = sorted(set(kw_senti_df['epoch'].tolist()))
            # senti for all models in array
            senti_values = []
            for sentiword_set in sentiword_set_arr:
                kw_senti_model_df = kw_senti_df[kw_senti_df['sentiword_set'] == sentiword_set]
                senti_values_for_model = []
                for epoch in epochs:
                    kw_senti_model_epoch_df = kw_senti_model_df[kw_senti_model_df['epoch'] == epoch]
                    value = kw_senti_model_epoch_df['value'].iloc[0]
                    senti_values_for_model.append(value)
                senti_values.append(senti_values_for_model)
            if include_expected:
                reference_values = senti_df['value'].tolist() if absolute else [item for sublist in senti_values for
                                                                                item in sublist]
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(kw) & expected_df['epoch'].isin(epochs)]
                expected_values = expected_kw_df['expected_valuation'].tolist()
                expected_transformed_values = expected_values if not with_axis else transform_expected_senti_values(
                    reference_values, expected_values)
            title = f'Wertungen des Schlagwortes {kw}'
            path = f'data/results/plots/senti/senti_{kw}_{"_".join(sentiword_set_arr)}_{"with_axis_" if with_axis else ""}' \
                   f'{"with_expected_" if include_expected else ""}{"abs_ref_" if absolute else ""}plot.png'
            epochs_df = pd.read_csv('data/epochs.csv')
            written_forms = epochs_df.loc[epochs_df['epoch_id'].isin(epochs), 'written_form'].tolist()
            # title
            plt.title(title)
            # prepare axes
            x = np.arange(len(epochs))
            ax = plt.gca()
            ax.set_xlim(0, len(epochs))
            plt.xticks(x, written_forms)
            plt.xlabel("Epochen")
            ylabel = 'Projektion auf die Achse der Ausgangswörter (pos: >0, neg: <0)'  if with_axis else 'durchschnittliche Ähnlichkeit mit den Senti-Word-Sets (pos: >0, neg: <0)'
            plt.ylabel(ylabel)
            plt.rc('grid', linestyle=':', color='black', linewidth=1)
            plt.axhline(0, color='black', linewidth=2)
            plt.grid(True)
            # plot expected values if wanted
            if include_expected:
                without_1000_vals = ignore_1000_vals(x, expected_transformed_values)
                x_segments = without_1000_vals[0]
                y_segments = without_1000_vals[1]
                # only plots expected values if without_1000_vals contains useful info
                if not all(not sublist for sublist in without_1000_vals):
                    plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', alpha=0.5, linewidth=70)
                    plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', linewidth=1,
                         label='expected values with error band')
            # plot senti
            colors = ['r', 'b', 'g', 'y', 'm']
            for a in range(0, len(senti_values)):
                plt.plot(x, senti_values[a], f'{colors[a]}-', label=sentiword_set_arr[a])
            # show legend
            plt.legend()
            # set tight layout (so that nothing is cut out)
            plt.tight_layout()
            # save diagram
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            fig.savefig(path)
            plt.close(fig)
            print(f"plot for {kw} saved")


# Connotations
# Code copied from https://github.com/ezosa/Diachronic-Embeddings/blob/master/embeddings_drift_tsne.py
# label points with words
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if i % 2 == 0:
            ax.text(point['x'] + .02, point['y'], str(point['val']))
        else:
            ax.text(point['x'] + .02, point['y'] - .02, str(point['val']))


# TODO: improve representation so that it becomes more meaningful
# Code copied from https://github.com/ezosa/Diachronic-Embeddings/blob/master/embeddings_drift_tsne.py
# maybe not much more useful than lists of nearest neighbors :/
def plot_words_from_time_epochs_tsne(epochs, target_word, aligned_base_folder, k=15, perplexity=30, mode_gensim=True, keep_doubles=True, iter=1000):
    # TODO: take care that legend shows epochs in correct order!
    # plot target word across all timeslices
    print("\nPlotting target word...")
    print("Target word: ", target_word)
    target_vectors = {}
    for epoch in epochs:
        epoch_written = utils.get_epoch_written_form_short(epoch)
        model_wv_path = f'{aligned_base_folder}/epoch{epoch}_lemma_200d_7w_cbow_aligned.wordvectors'
        # print("Year: ", model_years[year_index])
        word_vectors = KeyedVectors.load(model_wv_path)
        vocab = list(word_vectors.index_to_key)
        target_word_year = f'{target_word}_{epoch_written}'
        target_vectors[target_word_year] = {}
        target_vectors[target_word_year]['vector'] = word_vectors[target_word]
        target_vectors[target_word_year]['type'] = 'target_word'
        if mode_gensim:
            word_sim = word_vectors.most_similar(positive=target_word, topn=k)
        else:
            target_word_vec = [word_vectors[target_word]]
            vocab_sim = [cosine_similarity(target_word_vec, [word_vectors[vocab_word]]) for vocab_word in vocab if
                         vocab_word != target_word]
            word_sim = [(w, s) for s, w in sorted(zip(vocab_sim, vocab), reverse=True)][:k]
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
    words_to_plot = list(target_vectors.keys())
    len_words = len(words_to_plot)
    if len_words > 2:
        print("words to plot:", len_words)
        print(words_to_plot)
        vectors = [target_vectors[w]['vector'] for w in words_to_plot]
        word_types = [target_vectors[w]['type'] for w in words_to_plot]
        df = {'words': words_to_plot, 'type': word_types}
        df = pd.DataFrame.from_dict(df)
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=iter, learning_rate=100.0)
        tsne_results = tsne.fit_transform(np.array(vectors))
        print('t-SNE done!')
        df['tsne-one'] = tsne_results[:, 0]
        df['tsne-two'] = tsne_results[:, 1]
        plt.clf()
        plt.figure(figsize=(16, 10))
        n_colors = len(list(set(df['type'])))
        ax = sns.scatterplot(
            x='tsne-one', y='tsne-two',
            hue='type', s=50,
            palette=sns.color_palette("hls", n_colors),
            data=df,
            legend='full',
            alpha=1.0
        )
        label_point(df['tsne-one'], df['tsne-two'], df['words'], plt.gca())
        # draw lines between target words from different time points
        df_target = df[df['type'] == 'target_word']
        nrows = df_target.shape[0]
        lines = []
        for row_num in range(nrows-1):
            # draw line from time t to time t+1
            row1 = df_target.iloc[row_num, :]
            row2 = df_target.iloc[row_num+1, :]
            p1 = (row1['tsne-one'], row1['tsne-two'])
            p2 = (row2['tsne-one'], row2['tsne-two'])
            lines.append([p1, p2])
        lc = mc.LineCollection(lines, linewidths=1)
        ax.add_collection(lc)
        fig = ax.get_figure()
        fig.savefig(f'data/results/plots/associations/tsne_{target_word}_{"_".join(map(str, epochs))}_perpl{perplexity}_k{k}_{"gensim" if mode_gensim else "sklearn"}{"_doubles" if keep_doubles else ""}_steps_{iter}.png')
        plt.close()


def plot_tsne_according_to_occurrences(k=15, perplexity=30, mode_gensim=True, keep_doubles=True, iter=1000):
    # iterate rows in keywords_merged.csv
    df = pd.read_csv('data/keywords_merged.csv')
    for index, row in df.iterrows():
        # if word never occurs, ignore
        if row.first_occ_epoch != 0:
            # check if resp. start_epoch-folder exists
            aligned_base_folder = f'data/models/aligned_models/start_epoch_{row.first_occ_epoch}{f"_lh_{row.loophole}" if not str(0) in row.loophole else ""}'
            if os.path.isdir(aligned_base_folder):
                necessary_epochs = [item for item in range(row.first_occ_epoch, row.last_occ_epoch + 1) if str(item) not in row.loophole]
                plot_words_from_time_epochs_tsne(necessary_epochs, row.keyword, aligned_base_folder, k, perplexity, mode_gensim, keep_doubles, iter)


def plot_cosine_development_each_word():
    distances_df = pd.read_csv('data/results/cosine_developments.csv')
    # epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    keywords = keywords_df['keyword'].tolist()
    for kw in keywords:
        # ignore words that do not appear often enough
        if keywords_df[keywords_df['keyword'] == kw].ignore.iloc[0] == 0 and kw != "Wirtschaftsasylant":
            kw_distances_df = distances_df[distances_df['keyword'] == kw]
            title = f'Entwicklung von {kw}'
            path = f'data/results/plots/associations/dist_development_{kw}_plot.png'
            epochs_df = pd.read_csv('data/epochs.csv')
            first_epochs = kw_distances_df['first_epoch'].tolist()
            last_epoch = kw_distances_df['next_epoch'].tolist()[-1]
            epochs = first_epochs + [last_epoch]
            written_forms = [epochs_df[epochs_df['epoch_id'] == epoch]['written_form_short'].iloc[0] for epoch in epochs]
            distances = kw_distances_df['distance'].tolist()
            # title
            plt.title(title)
            # prepare axes
            ax = plt.gca()
            ax.set_xlim(0, len(epochs))
            plt.xticks(epochs, written_forms)
            plt.xlabel("Epochen")
            plt.ylabel('Kosinus-Abstand zwischen dem Wort in den jeweiligen Epochen')
            plt.rc('grid', linestyle=':', color='black', linewidth=1)
            plt.grid(True)
            # prepare data for plotting, making horizontal lines that are connected
            x = [((epochs[i] + epochs[i + 1])/2) for i in range(len(epochs) - 1)]
            x = x + [(epochs[i] + 0.1) for i in range(len(epochs) - 1)]
            x = x + [((epochs[i + 1]) - 0.1) for i in range(len(epochs) - 1)]
            x.sort()
            distances = [val for val in distances for _ in range(3)]
            # Create a new set of x values with more points for smoother lines
            x_smooth = np.linspace(min(x), max(x), 300)
            # linear interpolation
            y_smooth = np.interp(x_smooth, x, distances)
            ax.plot(x_smooth, y_smooth, color='blue', solid_capstyle='round', solid_joinstyle='round')
            # set tight layout (so that nothing is cut out)
            plt.tight_layout()
            # save diagram
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            fig.savefig(path)
            plt.close(fig)
            print(f"plot for {kw} saved")


def plot_exemplary_comparisons():
    # TODO: maybe move all experimental calculations to experiment.py
    # retrieve keywords that should be compared
    df = pd.read_csv('data/keyword_comparing.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    # iterate these compare groups
    for index, row in df.iterrows():
        # from main word, calculate the similarity for each other word for each epoch
        main_word = row.main_word
        # TODO: other words könnten ja auch nicht-keywords sein, die ich einfach gerne aus dem Pool des Wissens heraus testen würde!
        other_words = [row[column] for column in ['second_word', 'third_word', 'fourth_word', 'fifth_word'] if pd.notna(row[column])]
        # get necessary epochs
        kw_row = keywords_df[keywords_df['keyword'] == main_word].iloc[0]
        necessary_epochs = [item for item in range(kw_row.first_occ_epoch, kw_row.last_occ_epoch + 1) if
                            str(item) not in kw_row.loophole]
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
        # plot as many lines as other words
        title = f'Entwicklung im Verhältnis zum Schlagwort {main_word}'
        path = f'data/results/plots/associations/comparing_development_{main_word}_{"_".join(other_words)}_plot.png'
        filtered_epochs_df = epochs_df[epochs_df['epoch_id'].isin(necessary_epochs)]
        written_forms = filtered_epochs_df['written_form'].tolist()
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(necessary_epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(necessary_epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel(f'Cosine Similarity zum Wort {main_word}')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.grid(True)
        colors = ['r', 'b', 'g', 'y']
        # plot similarities
        for i in range(0, len(other_words)):
            plt.plot(x, results[i], f'{colors[i]}-', label=other_words[i])
        # plot legend
        plt.legend()
        # set tight layout (so that nothing is cut out)
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot saved")


# words not given in corpus:
# Vertriebener
# Geflüchteter
# Flüchtender
# Asylsuchender
# Asylberechtigter, Asylberechtigte

# spellings combined
# Heimatvertrieben, Heimatvertriebener --> 1
# Asylmißbrauch, Asylmissbrauch --> 2

# word freqs compared in one graph
# Fremdarbeiter, Gastarbeiter --> 3
# Flüchtling, Geflüchtete, Flüchtende --> 1
# Ausländer, Migrationshintergrund, Migrant --> 4
# Asylant, Asylbewerber --> 5
# Integration, multikulturell, Einwanderungsland --> 2
# evtl. noch Scheinasylant, Wirtschaftsasylant, Asylmiss[ß]brauch, Wirtschaftsflüchtling
# und evtl. noch Flüchtling, Asylant


# plot_words_from_time_epochs_tsne([1, 2], target_word='Flüchtling')
# plot_tsne_according_to_occurrences(k=10, perplexity=20)
# plot_sentiments(['standard', 'political'])
