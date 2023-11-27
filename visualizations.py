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


# TODO: keep semantic difference between "Heimatvertriebener" and "Heimatvertrieben"? Don't combine?
# Helper methods  # TODO: move to utils?
def skip_some_vals(x, y, val_to_skip=1000):
    x_segments = []
    y_segments = []
    current_segment_x = []
    current_segment_y = []
    for xi, yi in zip(x, y):
        if yi != val_to_skip:
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


def transform_expected_freqs_values(all_freqs, relevant_expected_values, exponential):
    transformed_values = []
    max_freqs = max(all_freqs)
    if not exponential:
        step_size = max_freqs/8
    else:
        relevant_expected_values = np.array(relevant_expected_values)
        relevant_expected_values = np.where(relevant_expected_values == 0, 0,
                                            np.where(relevant_expected_values >= 1000, 1000,
                                                     np.exp(relevant_expected_values)))
        step_size = max_freqs/np.exp(8)
    for val in relevant_expected_values:
        transformed_values.append(val if val == 1000 else step_size * val)
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
# TODO: eliminate warning(s) from exp. Or remove the exponential aspect.
def plot_frequencies(include_expected=True, absolute=False, exponential=False):
    freqs_df = pd.read_csv('data/results/freqs.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords2.csv')
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
    for i in range(len(keywords)):
        combine_group = combine_groups[i]
        # case 1: word has already been plotted
        if i in indices_done or ignoring[i]:
            continue
        # case 2: word does not need to be combined with other spelling and has not been plotted yet
        if not combine_group:
            kw = keywords[i]
            kw_freqs_df = freqs_df[freqs_df['keyword'] == kw]
            freqs = kw_freqs_df['pMW'].tolist()
            if include_expected:
                reference_values = freqs if not absolute else freqs_df['pMW'].tolist()
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(kw)]
                expected_values = expected_kw_df['expected_freq'].tolist()
                written_expected = expected_kw_df['written_freq'].tolist()
                # transform values using all existing freqs values
                transformed_exp_values = transform_expected_freqs_values(reference_values, expected_values, exponential)
            title = f'Häufigkeiten des Schlagwortes {kw}'
            path = f'data/results/plots/frequencies/freq_{kw}{"_with_expected" if include_expected else ""}{"_abs_ref" if absolute else ("_rel_ref" if include_expected else "")}{"_exp" if exponential else ""}_plot.png'
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
                kw_freqs = kw_freqs_df['pMW'].tolist()
                # save sum of all kw_freqs into freqs
                freqs = [sum(x) for x in zip(freqs, kw_freqs)]
            if include_expected:
                reference_values = freqs_df['pMW'].tolist() if absolute else freqs
                # get relevant info from expected_values for epochs and kw
                # it is assumed that combined words are in the same expectation group!
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(combined_kws[0])]
                expected_values = expected_kw_df['expected_freq'].tolist()
                written_expected = expected_kw_df['written_freq'].tolist()
                # transform values using all existing freqs values
                transformed_exp_values = transform_expected_freqs_values(reference_values, expected_values, exponential)
            indices_done.extend(combine_group_indices)
            title = f"Häufigkeiten der Schlagwörter {', '.join(combined_kws)}"
            path = f"data/results/plots/frequencies/freq_combined_{'_'.join(combined_kws)}{'_with_expected' if include_expected else ''}" \
                   f"{'_abs_ref' if absolute else ('_rel_ref' if include_expected else '')}{'_exp' if exponential else ''}_plot.png"
        written_forms = epochs_df['written_form'].tolist()  # TODO: assumes same order of epochs, maybe improve
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        # TODO: sieht z.B. bei Integration exp doof aus
        ax.set_ylim(0, max(freqs)+(max(freqs)*0.5))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('relative Häufigkeiten im Korpus in pMW')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.grid(True)
        # plot a grey area for the expected values
        if include_expected:
            without_1000_vals = skip_some_vals(x, transformed_exp_values)
            x_segments = without_1000_vals[0]
            y_segments = without_1000_vals[1]
            plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', alpha=0.5, linewidth=70)
            plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', linewidth=1,
                 label=f'erwartete Werte mit Error-Zone. Referenz: {"alle Häufigkeitswerte" if absolute else "Häufigkeitswerte des Wortes"}')
            # add labels to points
            for a, txt in enumerate(y_segments[0]):
                plt.text(x_segments[0][a], y_segments[0][a], f'{written_expected[x_segments[0][a]]}', ha='center', va='top', alpha=0.7)
        # plot freqs
        plt.plot(x, freqs, 'r-', label="Häufigkeiten in pMW (pro Million Wörter)")
        # add labels to points
        for a, txt in enumerate(freqs):
            plt.text(x[a], freqs[a], f'{freqs[a]:.2f}', ha='center', va='bottom')
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


def plot_comparing_frequencies():
    freqs_df = pd.read_csv('data/results/freqs.csv')
    df = pd.read_csv('data/keyword_comparing.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    # iterate compare groups
    for index, row in df.iterrows():
        # get all keywords in the row
        keywords = [item for item in row.tolist() if str(item) != 'nan']
        # limit senti_df to only these keywords and the wanted sentiword_model
        freqs_df_filtered = freqs_df[freqs_df['word'].isin(keywords)]
        # get the needed epochs
        epochs = sorted(set(freqs_df_filtered['epoch'].tolist()))
        # save values in a nested array
        freqs = []
        for w in keywords:
            freqs.append([])
            for epoch in epochs:
                val = freqs_df_filtered[(freqs_df_filtered['epoch'] == epoch) & (freqs_df_filtered['word'] == w)]['pMW'].iloc[0]
                freqs[keywords.index(w)].append(val)
        title = f"Häufigkeiten der Schlagwörter {', '.join(keywords)}"
        path = f"data/results/plots/frequencies/freq_compared_{'_'.join(keywords)}_plot.png"
        written_forms = epochs_df[epochs_df['epoch_id'].isin(epochs)]['written_form'].tolist()
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('relative Häufigkeiten im Korpus in pMW (pro Million Wörter)')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.grid(True)
        # plot freqs
        colors = ['r', 'b', 'g', 'y', 'm']
        for a in range(0, len(freqs)):
            plt.plot(x, freqs[a], f'{colors[a]}-', label=keywords[a])
            for o, txt in enumerate(freqs[a]):
                plt.text(x[o], freqs[a][o], f'{freqs[a][o]:.2f}', color=colors[a], ha='center', va='bottom')
        # show legend
        plt.legend()
        # set tight layout (so that nothing is cut out)
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {index} saved")


# TODO: needed? maybe remove?
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


def plot_frequency_distribution_for_corpora_keywords():
    df = pd.read_csv('data/results/freqs.csv')
    for epoch in range(1, 9):
        # dict = experiment.total_word_frequency_distribution(epoch)
        epoch_df = df[df['epoch'] == epoch]
        counts = epoch_df['pMW'].tolist()
        sorted_counts = sorted(counts, reverse=True)
        ranks = list(range(1, len(sorted_counts) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, np.log(sorted_counts), linestyle='-', color='b', alpha=0.5)
        plt.suptitle(f'Verteilung der Worthäufigkeit der Schlagwörter in Relation zum Rang, Epoche {epoch}', fontsize=16,
                     fontweight="bold")
        # plt.title(f'Gesamtgröße des Vokabulars: {len(ranks)} Wörter. X-Achse zur Übersichtlichkeit begrenzt')
        plt.xlabel('Rang')
        plt.ylabel('Häufigkeit')
        plt.grid(True)
        plt.xlim(0, len(ranks))
        fig = plt.gcf()
        fig.savefig(f'data/results/plots/frequencies/keyword_distribution_epoch{epoch}_log.png')
        plt.close(fig)
        print(f'epoch {epoch} plotted')


# TODO: in- or exclude normalization by freq in a meaningful way
# Sentiments/Valuation
def plot_sentiments(sentiword_set_arr, with_axis=False, include_expected=True, absolute=False):
    senti_df = pd.read_csv(f'data/results/senti{"_with_axis" if with_axis else ""}.csv')
    senti_df_filtered = senti_df[senti_df['sentiword_set'].isin(sentiword_set_arr)]
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    if include_expected:
        expected_df = pd.read_csv('data/expected_values.csv')
    # keywords = utils.load_keywords()
    keywords = keywords_df['keyword'].tolist()
    combine_groups = keywords_df['combine_group'].tolist()
    ignoring = keywords_df['ignore'].tolist()
    indices_done = []
    for i in range(0, len(keywords)):
        combine_group = combine_groups[i]
        # case 1: word has already been plotted or is not present in the whole corpus
        if i in indices_done or ignoring[i] == 1:
            continue
        # case 2: word does not need to be combined with other spelling and has not been plotted yet
        if combine_group == 0:
            kw = keywords[i]
            kw_senti_df = senti_df_filtered[senti_df_filtered['word'] == kw]
            # only take epochs that are valid
            epochs = sorted(set(kw_senti_df['epoch'].tolist()))
            # senti for all models in array
            senti_values = []
            for sentiword_set in sentiword_set_arr:
                kw_senti_model_df = kw_senti_df[kw_senti_df['sentiword_set'] == sentiword_set]
                senti_values_for_model = []
                for epoch in epochs:
                    # value = kw_senti_model_df[kw_senti_model_df['epoch'] == epoch]['value'].iloc[0]
                    value = kw_senti_model_df[kw_senti_model_df['epoch'] == epoch]['normalized_by_freq'].iloc[0]
                    senti_values_for_model.append(value)
                senti_values.append(senti_values_for_model)
            if include_expected:
                # reference_values = senti_df_filtered['value'].tolist() if absolute else [item for sublist in senti_values for
                #                                                                 item in sublist]
                reference_values = senti_df_filtered['normalized_by_freq'].tolist() if absolute else [item for sublist in senti_values for
                                                                                item in sublist]
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(kw) & expected_df['epoch'].isin(epochs)]
                expected_values = expected_kw_df['expected_valuation'].tolist()
                written_expected = expected_kw_df['written_senti'].tolist()
                expected_transformed_values = expected_values if absolute and not with_axis else transform_expected_senti_values(
                    reference_values, expected_values)
            indices_done.append(i)
            title = f'Wertungen des Schlagwortes {kw} anhand {"einer Polaritätsachse" if with_axis else "WEAT"}'
            path = f'data/results/plots/senti/senti_{kw}_{"_".join(sentiword_set_arr)}_{"with_axis_" if with_axis else "weat_"}{"abs_ref_" if absolute else ("rel_ref_" if include_expected else "")}normalized_plot.png'
        # case 3: combine different spellings
        else:
            # find all spellings for word
            combine_group_indices = [i for i, x in enumerate(combine_groups) if x == combine_group]
            # TODO: does this also work if we had multiple combine groups for a word?
            combined_kws = [keywords[cgi] for cgi in combine_group_indices]
            combined_kw_df = senti_df_filtered[senti_df_filtered['word'].isin(combined_kws)]
            # only take epochs that are needed
            epochs = sorted(set(combined_kw_df['epoch'].tolist()))
            senti_values = []
            for sentiword_set in sentiword_set_arr:
                combined_kw_senti_model_df = combined_kw_df[combined_kw_df['sentiword_set'] == sentiword_set]
                senti_values_for_model = []
                for epoch in epochs:
                    combined_kw_senti_model_epoch_df = combined_kw_senti_model_df[combined_kw_senti_model_df['epoch'] == epoch]
                    # get value as mean between all given values
                    # senti_value = np.mean(combined_kw_senti_model_epoch_df['value'].tolist())
                    senti_value = np.mean(combined_kw_senti_model_epoch_df['normalized_by_freq'].tolist())
                    senti_values_for_model.append(senti_value)
                senti_values.append(senti_values_for_model)
            if include_expected:
                # reference_values = senti_df_filtered['value'].tolist() if absolute else [item for sublist in senti_values for
                #                                                                 item in sublist]
                reference_values = senti_df_filtered['normalized_by_freq'].tolist() if absolute else [item for sublist in senti_values for
                                                                                item in sublist]
                # get relevant info from expected_values for epochs and kw
                # it is assumed that combined kws are in the same expectation group!
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(combined_kws[0]) & expected_df['epoch'].isin(epochs)]
                expected_values = expected_kw_df['expected_valuation'].tolist()
                written_expected = expected_kw_df['written_senti'].tolist()
                expected_transformed_values = expected_values if absolute and not with_axis else transform_expected_senti_values(
                    reference_values, expected_values)
            indices_done.extend(combine_group_indices)
            title = f'Wertungen der Schlagwörter {", ".join(combined_kws)} anhand {"einer Polaritätsachse" if with_axis else "WEAT"}'
            path = f"data/results/plots/senti/senti_combined_{'_'.join(combined_kws)}_{'_'.join(sentiword_set_arr)}_{'_'.join(sentiword_set_arr)}_{'with_axis_' if with_axis else 'weat_'}{'abs_ref_' if absolute else ('rel_ref_' if include_expected else '')}normalized_plot.png"
        epochs_df = pd.read_csv('data/epochs.csv')
        written_forms = epochs_df.loc[epochs_df['epoch_id'].isin(epochs), 'written_form'].tolist()  # TODO: retrieve written forms only for relevant epochs
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('Projektion auf die Polaritätsachse (pos: >0, neg: <0)' if with_axis else 'durchschnittliche Ähnlichkeit mit den Ausgangswörtern (pos: >0, neg: <0)')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.axhline(0, color='black', linewidth=2)
        plt.grid(True)
        # plot expected values if wanted
        if include_expected:
            without_1000_vals = skip_some_vals(x, expected_transformed_values)
            x_segments = without_1000_vals[0]
            y_segments = without_1000_vals[1]
            # only plots expected values if without_1000_vals contains useful info
            if not all(not sublist for sublist in without_1000_vals):
                plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', alpha=0.5, linewidth=70)
                plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', linewidth=1,
                     label=f'erwartete Werte mit Error-Zone. Referenz: {"alle Häufigkeitswerte" if absolute else "Häufigkeitswerte des Wortes"}')
            # add labels to points
            for a, txt in enumerate(y_segments[0]):
                plt.text(x_segments[0][a], y_segments[0][a], f'{written_expected[x_segments[0][a]]}', ha='center', va='top', alpha=0.7)
        # plot senti
        colors = ['r', 'b', 'g', 'y', 'm']
        for a in range(0, len(senti_values)):
            plt.plot(x, senti_values[a], f'{colors[a]}-', label=sentiword_set_arr[a])
            # add labels to points
            for o, txt in enumerate(senti_values[a]):
                plt.text(x[o], senti_values[a][o], f'{senti_values[a][o]:.2f}', ha='center', va='bottom')
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


def plot_comparing_sentiments(sentiword_set="standard", with_axis=True):
    filename = f'data/results/{"senti.csv" if not with_axis else "senti_with_axis.csv"}'
    senti_df = pd.read_csv(filename)
    senti_model_df = senti_df[senti_df['sentiword_set'] == sentiword_set]
    df = pd.read_csv('data/keyword_comparing.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    # iterate compare groups
    for index, row in df.iterrows():
        # get all keywords in the row
        keywords = [item for item in row.tolist() if str(item) != 'nan']
        # limit senti_df to only these keywords and the wanted sentiword_model
        senti_df_filtered = senti_model_df[senti_model_df['word'].isin(keywords)]
        # get the needed epochs
        epochs = sorted(set(senti_df_filtered['epoch'].tolist()))
        # save values in a nested array, taking care of None values, too
        senti = []
        # doing this, also take care of cases where one of the words does not exist in epoch
        for w in keywords:
            senti.append([])
            for epoch in epochs:
                try:
                    # val = senti_df_filtered[(senti_df_filtered['epoch'] == epoch) & (senti_df_filtered['word'] == w)]['value'].iloc[0]
                    val = senti_df_filtered[(senti_df_filtered['epoch'] == epoch) & (senti_df_filtered['word'] == w)]['normalized_by_freq'].iloc[0]
                except IndexError:
                    val = None
                senti[keywords.index(w)].append(val)
        # TODO: suptitle too long! Also, fix space to title
        suptitle = f'Wertungen der Schlagwörter {", ".join(keywords)}'
        title = f'anhand {"einer Polaritätsachse" if with_axis else "WEAT"} & {sentiword_set} Wort Set'
        path = f"data/results/plots/senti/senti_compared_{'_'.join(keywords)}_{sentiword_set}{'_with_axis' if with_axis else '_weat'}_normalized_plot.png"
        written_forms = epochs_df[epochs_df['epoch_id'].isin(epochs)]['written_form'].tolist()
        # titles
        plt.suptitle(suptitle, fontsize=16, fontweight="bold")
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('Projektion auf die Polaritätsachse (pos: >0, neg: <0)' if with_axis else 'durchschnittliche Ähnlichkeit mit den Ausgangswörtern (pos: >0, neg: <0)')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.axhline(0, color='black', linewidth=2)
        plt.grid(True)
        # plot senti
        colors = ['r', 'b', 'g', 'y', 'm']
        for a in range(0, len(senti)):
            plt.plot(x, senti[a], f'{colors[a]}-', label=keywords[a])
            for o, txt in enumerate(senti[a]):
                if senti[a][o]:
                    plt.text(x[o], senti[a][o], f'{senti[a][o]:.2f}', color=colors[a], ha='center', va='bottom')
        # show legend
        plt.legend()
        # set tight layout (so that nothing is cut out)
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {index} saved")


# Connotations
# Code copied from https://github.com/ezosa/Diachronic-Embeddings/blob/master/embeddings_drift_tsne.py
# label points with words
def label_point(x, y, val, type, sim, ax):
    a = pd.concat({'x': x, 'y': y, 'type': type, 'sim': sim, 'val': val}, axis=1)
    for i, point in a.iterrows():
        font_size = 16 if point['type'] == 'target_word' else (15 if point['sim'] > 0.79 else (14 if point['sim'] > 0.69
                                                                                               else (13 if point['sim'] > 0.59
                                                                                                     else 12)))
        weight = 'semibold' if point['type'] == 'target_word' else 'normal'
        ax.text(point['x'] + .02, point['y'] - .02 if i % 2 == 0 else point['y'], str(point['val']), size=font_size,
                    weight=weight)


# TODO: maybe remove sklearn.cosine_similarity! Then scikit-learn can be left out
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
        sims = [target_vectors[w]['sim'] if 'sim' in target_vectors[w] else 100.0 for w in words_to_plot]
        df = {'words': words_to_plot, 'type': word_types, 'sim': sims}
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
        label_point(df['tsne-one'], df['tsne-two'], df['words'], df['type'], df['sim'], plt.gca())
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
