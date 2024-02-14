import experiment
import utils
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE  # needed to be installed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # needed to be installed
from matplotlib import collections as mc
import os
import math
from wordcloud import WordCloud


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


def transform_expected_freqs_values(all_freqs, relevant_expected_values, relative):
    transformed_values = []
    if relative:
        max_freqs = max(all_freqs)
        step_size = max_freqs/8
        for val in relevant_expected_values:
            transformed_values.append(val if val == 1000 else step_size * val)
    else:
        # get slice values for expected values from csv
        slice_info_df = pd.read_csv('data/results/expected_freq_results_slices.csv')
        for val in relevant_expected_values:
            transformed_values.append(val if val == 1000 or val == 0 else
                                      slice_info_df[slice_info_df['expected_freq_key'] == val][
                                          'pMW_mean'].iloc[0])
    return transformed_values


def transform_expected_senti_values(expected_values, sentiword_set):
    transformed_values = []
    slice_info_df = pd.read_csv('data/results/expected_senti_results_slices.csv')
    for val in expected_values:
        transformed_values.append(val if val == 1000 else
                                  slice_info_df[(slice_info_df['expected_senti_key'] == val) &
                                                (slice_info_df['sentiword_set'] == sentiword_set)]['senti_mean'].iloc[0])
    return transformed_values


# Frequency methods
def plot_frequencies(include_expected=True, relative=False, show_result_groups=True):
    """
    plot frequency values that have been calculated and saved in csv in experiment.save_frequency_info_in_csv
    :param include_expected: True if the expected values should be plotted as a reference
    :param relative: True if expected values should be plotted as relative to the values of the word only,
    False if all frequency values (grouped into 8 slices) should be taken as a reference
    experiment.make_freq_slices should have been executed as a prerequisite if False
    :param show_result_groups: True if colors in the background should indicate the groups of values (high/low).
    experiment.make_freq_slices should have been executed as a prerequisite
    :return: (save line plots in data/results/plots/frequencies)
    """
    freqs_df = pd.read_csv('data/results/freqs.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    # keywords_df = pd.read_csv('data/keywords2.csv')
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
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(kw)]
                expected_values = expected_kw_df['expected_freq'].tolist()
                written_expected = expected_kw_df['written_freq'].tolist()
                # transform values using all existing freqs values
                transformed_exp_values = transform_expected_freqs_values(freqs, expected_values, relative)
            title = f'Häufigkeiten des Schlagwortes {kw}'
            path = f'data/results/plots/frequencies/freq_{kw}{"_res_groups" if show_result_groups else ""}{"_incl_exp" if include_expected else ""}{"_abs_ref" if not relative else ("_rel_ref" if include_expected else "")}_plot.png'
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
                # get relevant info from expected_values for epochs and kw
                # it is assumed that combined words are in the same expectation group!
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(combined_kws[0])]
                expected_values = expected_kw_df['expected_freq'].tolist()
                written_expected = expected_kw_df['written_freq'].tolist()
                # transform values using all existing freqs values
                transformed_exp_values = transform_expected_freqs_values(freqs, expected_values, relative)
            indices_done.extend(combine_group_indices)
            title = f"Häufigkeiten der Schlagwörter {', '.join(combined_kws)}"
            path = f"data/results/plots/frequencies/freq_combined_{'_'.join(combined_kws)}{'_res_groups' if show_result_groups else ''}{'_incl_exp' if include_expected else ''}" \
                   f"{'_abs_ref' if not relative else ('_rel_ref' if include_expected else '')}_plot.png"
        written_forms = epochs_df['written_form'].tolist()  # TODO: assumes same order of epochs, maybe improve
        # title
        plt.title(title)
        # prepare axes
        x = np.arange(len(epochs))
        ax = plt.gca()
        ax.set_xlim(0, len(epochs))
        plt.xticks(x, written_forms)
        plt.xlabel("Epochen")
        plt.ylabel('relative Häufigkeiten im Korpus in pMW')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.grid(True)
        max_y = max(freqs)
        # plot a grey area for the expected values
        if include_expected:
            without_1000_vals = skip_some_vals(x, transformed_exp_values)
            x_segments = without_1000_vals[0]
            y_segments = without_1000_vals[1]
            max_y = max(max_y, max(y_segments[0]))
            # plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', alpha=0.5, linewidth=70)
            plt.plot(x_segments[0], y_segments[0], color='dimgrey', linestyle='-', linewidth=1,
                 label=f'erwartete Werte, Referenz: {"alle Häufigkeitswerte" if not relative else "Häufigkeitswerte des Wortes"}')
            # add labels to points
            if not relative:
                for a, txt in enumerate(y_segments[0]):
                    plt.text(x_segments[0][a], y_segments[0][a], f'{written_expected[x_segments[0][a]]}', ha='center', va='top', alpha=0.7)
        # plot freqs
        plt.plot(x, freqs, 'r-', label="Häufigkeiten in pMW (pro Million Wörter)")
        # add labels to points
        for a, txt in enumerate(freqs):
            plt.text(x[a], freqs[a], f'{freqs[a]:.2f}', color='red', ha='center', va='bottom')
        if show_result_groups:
            # plot blue areas to give a hint on the dimensions
            colors = plt.cm.Blues(np.linspace(0.2, 0.9, 8))
            slice_info_df = pd.read_csv('data/results/expected_freq_results_slices.csv')
            all_exp_df = pd.read_csv('data/expected_freq_translation.csv')
            for index in range(1, 9):
                max_slice = slice_info_df[slice_info_df['expected_freq_key'] == index]['pMW_max'].iloc[0]
                min_slice = slice_info_df[slice_info_df['expected_freq_key'] == index]['pMW_min'].iloc[0]
                mean_slice = slice_info_df[slice_info_df['expected_freq_key'] == index]['pMW_mean'].iloc[0]
                ax.axhspan(min_slice, max_slice, facecolor=colors[index - 1], alpha=0.35)
                if min(freqs) <= max_slice or max(freqs) <= max_slice:
                    plt.text(0.1, mean_slice,
                             f'{all_exp_df[all_exp_df["freq_value"] == index]["written"].values[0]}',
                             color=colors[index - 1], fontsize=10)
        ax.set_ylim([0-max_y*0.02, max_y + 0.1*max_y])
        # plot legend
        plt.legend()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {i} saved")


def plot_comparing_frequencies(show_result_groups=True):
    """
        plot frequency values of multiple words to enable comparing them. Values have been calculated and saved in csv
        in experiment.save_frequency_info_in_csv
        :param show_result_groups: True if colors in the background should indicate the groups of values (high/low).
        experiment.make_freq_slices should have been executed as a prerequisite
        :return: (save line plots in data/results/plots/frequencies)
    """
    freqs_df = pd.read_csv('data/results/freqs.csv')
    df = pd.read_csv('data/keyword_comparing.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    # iterate compare groups
    for i, row in df.iterrows():
        # get all keywords in the row
        keywords = [item for item in row.tolist() if str(item) != 'nan']
        # limit senti_df to only these keywords and the wanted sentiword_model
        freqs_df_filtered = freqs_df[freqs_df['keyword'].isin(keywords)]
        # get the needed epochs
        epochs = sorted(set(freqs_df_filtered['epoch'].tolist()))
        # save values in a nested array
        freqs = []
        for w in keywords:
            freqs.append([])
            for epoch in epochs:
                val = freqs_df_filtered[(freqs_df_filtered['epoch'] == epoch) & (freqs_df_filtered['keyword'] == w)]['pMW'].iloc[0]
                freqs[keywords.index(w)].append(val)
        title = f"Häufigkeiten der Schlagwörter {', '.join(keywords)}"
        path = f"data/results/plots/frequencies/freq_compared_{'_'.join(keywords)}{'_res_groups' if show_result_groups else ''}_plot.png"
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
        max_y = max([max(i) for i in freqs])
        if show_result_groups:
            # plot blue areas to give a hint on the dimensions
            blue_colors = plt.cm.Blues(np.linspace(0.2, 0.9, 8))
            slice_info_df = pd.read_csv('data/results/expected_freq_results_slices.csv')
            all_exp_df = pd.read_csv('data/expected_freq_translation.csv')
            for index in range(1, 9):
                max_slice = slice_info_df[slice_info_df['expected_freq_key'] == index]['pMW_max'].iloc[0]
                min_slice = slice_info_df[slice_info_df['expected_freq_key'] == index]['pMW_min'].iloc[0]
                mean_slice = slice_info_df[slice_info_df['expected_freq_key'] == index]['pMW_mean'].iloc[0]
                ax.axhspan(min_slice, max_slice, facecolor=blue_colors[index - 1], alpha=0.35)
                if min([min(i) for i in freqs]) <= max_slice or max_y <= max_slice:
                    plt.text(0.1, mean_slice,
                             f'{all_exp_df[all_exp_df["freq_value"] == index]["written"].values[0]}',
                             color=blue_colors[index - 1], fontsize=10)
        ax.set_ylim([0-max_y*0.02, max_y + 0.1*max_y])
        # show legend
        plt.legend()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {i} saved")


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


# TODO: delete
def plot_mean_frequencies_as_bar_plot():
    df = pd.read_csv('data/results/mean_freqs.csv')
    df = df.sort_values(by=['rank'], ascending=False)
    words = df['keyword'].tolist()
    ranks = df['rank'].tolist()
    words_with_ranks = [f'{ranks[i]}. {words[i]}' for i in range(len(words))]
    freqs = df['mean_freq'].tolist()
    # creating the bar plot
    plt.barh(words_with_ranks, freqs, color='maroon',
            height=0.5)
    for i in range(len(words_with_ranks)):
        plt.text(freqs[i], i, f'  {math.ceil(freqs[i] * 10) / 10}', va='center')
    plt.ylabel("Schlagwörter")
    plt.xlabel("Durchschnittliche Häufigkeit in pMW")
    plt.title("Durchschnittliche Häufigkeiten der Schlagwörter über alle Epochen, \n1949-2023")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    fig.savefig('data/results/plots/frequencies/mean_freqs.png')
    plt.close(fig)


def plot_mean_frequencies_over_epochs_as_bar_plot():
    results = experiment.calculate_mean_frequency_over_all_keywords()
    freqs = [results[i].get('mean_freq') for i in range(len(results))]
    epochs = [results[i].get('epoch') for i in range(len(results))]
    # creating the bar plot
    plt.bar(epochs, freqs, color='maroon')
    # for i in range(len(words_with_ranks)):
    # plt.text(freqs[i], i, f'  {math.ceil(freqs[i] * 10) / 10}', va='center')
    plt.ylabel("durchschnittliche Häufigkeit der Schlagwörter")
    plt.xlabel("Epochen")
    plt.title("Durchschnittliche Häufigkeiten aller Schlagwörter in den jeweiligen Epochen, \n1949-2023")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    fig.savefig('data/results/plots/frequencies/mean_freqs_for_epochs.png')
    plt.close(fig)


def plot_frequency_maxima_for_epochs_as_bar_plot():
    maxima = experiment.get_freq_maxima_for_epochs()
    # creating the bar plot
    plt.bar([entry['name'] for entry in maxima.values()], [entry['count'] for entry in maxima.values()], color='maroon')
    plt.ylabel("Anzahl der Häufigkeits-Maxima")
    plt.xlabel("Epochen")
    plt.title("Häufigkeits-Maxima aller Schlagwörter in den jeweiligen Epochen, \n1949-2023")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    fig.savefig('data/results/plots/frequencies/freqs_maxima.png')
    plt.close(fig)


# Sentiments/Valuation
def plot_sentiments(sentiword_set_arr, include_expected=True, show_result_groups=True, with_axis=False):
    """
    plot sentiment values that have been calculated and saved in csv in experiment.analyse_senti_valuation_of_keywords
    :param sentiword_set_arr: one or more of 'political', 'standard' and 'combination' (comprising the mean of the two previous)
    :param include_expected: True if the expected values should be plotted as a reference
    :param show_result_groups: True if colors in the background should indicate the groups of values (high/low).
    experiment.make_senti_slices should have been executed as a prerequisite
    :param with_axis: True if the polarity axis method should be used instead of WEAT
    :return: (save line plots in data/results/plots/senti)
    """
    senti_df = pd.read_csv(f'data/results/senti{"_with_axis" if with_axis else ""}.csv')
    senti_df_filtered = senti_df[senti_df['sentiword_set'].isin(sentiword_set_arr)]
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    # keywords_df = pd.read_csv('data/keywords2.csv')
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
                    value = kw_senti_model_df[kw_senti_model_df['epoch'] == epoch]['value'].iloc[0]
                    senti_values_for_model.append(value)
                senti_values.append(senti_values_for_model)
            if include_expected:
                # get relevant info from expected_values for epochs and kw
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(kw) & expected_df['epoch'].isin(epochs)]
                expected_values = expected_kw_df['expected_senti'].tolist()
                written_expected = expected_kw_df['written_senti'].tolist()
                expected_transformed_values = transform_expected_senti_values(expected_values, sentiword_set_arr[0])
            indices_done.append(i)
            title = f'Wertungen des Schlagwortes {kw} anhand {"einer Polaritätsachse" if with_axis else "WEAT"}'
            path = f'data/results/plots/senti/senti_{kw}_{"_".join(sentiword_set_arr)}_{"with_axis_" if with_axis else "weat_"}{"res_groups_" if show_result_groups else ""}{"incl_exp_" if include_expected else ""}plot.png'
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
                    senti_value = np.mean(combined_kw_senti_model_epoch_df['value'].tolist())
                    senti_values_for_model.append(senti_value)
                senti_values.append(senti_values_for_model)
            if include_expected:
                # get relevant info from expected_values for epochs and kw
                # it is assumed that combined kws are in the same expectation group!
                # the first sentiword_arr is taken as a reference for the transformed values if they are adapted
                expected_kw_df = expected_df[expected_df['keyword'].str.contains(combined_kws[0]) & expected_df['epoch'].isin(epochs)]
                expected_values = expected_kw_df['expected_senti'].tolist()
                written_expected = expected_kw_df['written_senti'].tolist()
                expected_transformed_values = transform_expected_senti_values(expected_values, sentiword_set_arr[0])
            indices_done.extend(combine_group_indices)
            title = f'Wertungen der Schlagwörter {", ".join(combined_kws)} anhand {"einer Polaritätsachse" if with_axis else "WEAT"}'
            path = f"data/results/plots/senti/senti_combined_{'_'.join(combined_kws)}_{'_'.join(sentiword_set_arr)}_{'with_axis_' if with_axis else 'weat_'}{'res_groups_' if show_result_groups else ''}{'incl_exp_' if include_expected else ''}plot.png"
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
        plt.ylabel('Projektion auf die Polaritätsachse (pos: >0, neg: <0)' if with_axis else 'durchschnittliche Ähnlichkeit mit den Ausgangswörtern (pos: >0, neg: <0)')
        plt.rc('grid', linestyle=':', color='black', linewidth=1)
        plt.axhline(0, color='black', linewidth=2)
        plt.grid(True)
        min_y = None
        max_y = None
        # plot expected values if wanted
        if include_expected:
            without_1000_vals = skip_some_vals(x, expected_transformed_values)
            x_segments = without_1000_vals[0]
            y_segments = without_1000_vals[1]
            min_y = min(y_segments[0])
            max_y = max(y_segments[0])
            # only plots expected values if without_1000_vals contains useful info
            if not all(not sublist for sublist in without_1000_vals):
                # plt.plot(x_segments[0], y_segments[0], color='gray', linestyle='-', alpha=0.5, linewidth=70)
                plt.plot(x_segments[0], y_segments[0], color='dimgrey', linestyle='-', linewidth=1,
                         label='erwartete Werte')
            # add labels to points
            for a, txt in enumerate(y_segments[0]):
                plt.text(x_segments[0][a], y_segments[0][a], f'{written_expected[x_segments[0][a]]}', ha='center', va='top', alpha=0.7)
        # plot senti
        colors = ['r', 'b', 'g', 'y', 'm']
        for a in range(0, len(senti_values)):
            plt.plot(x, senti_values[a], f'{colors[a]}-', label=f'WEAT-Wert nach Ausgangswörtern: {sentiword_set_arr[a] if sentiword_set_arr[a] != "combination" else "political & standard"}')
            # add labels to points
            for o, txt in enumerate(senti_values[a]):
                plt.text(x[o], senti_values[a][o], f'{senti_values[a][o]:.2f}', color=colors[a], ha='center', va='bottom')
        # plot red/green areas to give a hint on the dimensions
        min_val = min([min(val) for val in senti_values])
        max_val = max([max(val) for val in senti_values])
        min_y = min(min_y, min_val) if min_y is not None else min_val
        max_y = max(max_y, max_val) if max_y is not None else max_val
        if show_result_groups:
            red_colors = plt.cm.Reds(np.linspace(0.9, 0.2, 4))
            green_colors = plt.cm.Greens(np.linspace(0.2, 0.9, 4))
            senti_colors = np.concatenate([red_colors, np.array([[1.0, 1.0, 1.0, 1.0]]), green_colors])
            slice_info_df = pd.read_csv('data/results/expected_senti_results_slices.csv')
            all_exp_df = pd.read_csv('data/expected_senti_translation.csv')
            slice_info_sentiword_set_df = slice_info_df[slice_info_df['sentiword_set'] == sentiword_set]
            for senti_key in slice_info_sentiword_set_df['expected_senti_key'].tolist():
                index = slice_info_sentiword_set_df['expected_senti_key'].tolist().index(senti_key)
                max_slice = slice_info_sentiword_set_df[slice_info_sentiword_set_df['expected_senti_key'] == senti_key]['senti_max'].iloc[0]
                min_slice = slice_info_sentiword_set_df[slice_info_sentiword_set_df['expected_senti_key'] == senti_key]['senti_min'].iloc[0]
                mean_slice = slice_info_sentiword_set_df[slice_info_sentiword_set_df['expected_senti_key'] == senti_key]['senti_mean'].iloc[0]
                ax.axhspan(min_slice, max_slice, facecolor=senti_colors[index], alpha=0.35)
                if min_y <= max_slice or max_y <= max_slice:
                    plt.text(0.1, mean_slice,
                             f'{all_exp_df[all_exp_df["senti_value"] == senti_key]["written"].values[0]}',
                             color=senti_colors[index], fontsize=10)
        ax.set_ylim([min_y - 0.025, max_y + 0.025])
        # show legend
        plt.legend()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {i} saved")


def plot_comparing_sentiments(sentiword_set="combination", show_result_groups=True, with_axis=False):
    """
    plot sentiment values of multiple words to enable comparing them. Values have been calculated and saved in csv in
    experiment.analyse_senti_valuation_of_keywords
    :param sentiword_set: one of 'political', 'standard' and 'combination' (comprising the mean of the two previous)
    :param show_result_groups: True if colors in the background should indicate the groups of values (high/low)
    :param with_axis: True if the polarity axis method should be used instead of WEAT
    :return: (save line plots in data/results/plots/senti)
    """
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
                    val = senti_df_filtered[(senti_df_filtered['epoch'] == epoch) & (senti_df_filtered['word'] == w)]['value'].iloc[0]
                except IndexError:
                    val = None
                senti[keywords.index(w)].append(val)
        suptitle = f'Wertungen der Schlagwörter \n{", ".join(keywords)}'
        title = f'anhand {"einer Polaritätsachse" if with_axis else "WEAT"} & {sentiword_set} Wort Set'
        path = f"data/results/plots/senti/senti_compared_{'_'.join(keywords)}_{sentiword_set}{'_with_axis' if with_axis else '_weat'}{'_res_groups' if show_result_groups else ''}_plot.png"
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
        colors = ['r', 'b', 'g', 'm', 'c']
        for a in range(0, len(senti)):
            plt.plot(x, senti[a], f'{colors[a]}-', label=keywords[a])
            for o, txt in enumerate(senti[a]):
                if senti[a][o]:
                    plt.text(x[o], senti[a][o], f'{senti[a][o]:.2f}', color=colors[a], ha='center', va='bottom')
        min_y = min([min(value for value in sublist if value is not None) for sublist in senti])
        max_y = max([max(value for value in sublist if value is not None) for sublist in senti])
        # plot red/green areas to give a hint on the dimensions
        if show_result_groups:
            red_colors = plt.cm.Reds(np.linspace(0.9, 0.2, 4))
            green_colors = plt.cm.Greens(np.linspace(0.2, 0.9, 4))
            senti_colors = np.concatenate([red_colors, np.array([[1.0, 1.0, 1.0, 1.0]]), green_colors])
            slice_info_df = pd.read_csv('data/results/expected_senti_results_slices.csv')
            slice_info_df = slice_info_df[slice_info_df['sentiword_set'] == sentiword_set]
            all_exp_df = pd.read_csv('data/expected_senti_translation.csv')
            for senti_key in slice_info_df['expected_senti_key'].tolist():
                i = slice_info_df['expected_senti_key'].tolist().index(senti_key)
                max_slice = slice_info_df[slice_info_df['expected_senti_key'] == senti_key]['senti_max'].iloc[0]
                min_slice = slice_info_df[slice_info_df['expected_senti_key'] == senti_key]['senti_min'].iloc[0]
                mean_slice = slice_info_df[slice_info_df['expected_senti_key'] == senti_key]['senti_mean'].iloc[0]
                ax.axhspan(min_slice, max_slice, facecolor=senti_colors[i], alpha=0.35)
                if min_y <= max_slice or max_y <= max_slice:
                    plt.text(0.1, mean_slice,
                             f'{all_exp_df[all_exp_df["senti_value"] == senti_key]["written"].values[0]}',
                             color=senti_colors[i], fontsize=10)
        ax.set_ylim([min_y - 0.025, max_y + 0.025])
        # show legend
        plt.legend()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot {index} saved")


def plot_senti_distribution_():
    df = pd.read_csv('data/results/senti.csv')
    filtered_df = df[df['sentiword_set'] == 'combination']
    counts = filtered_df['value'].tolist()
    sorted_counts = sorted(counts, reverse=True)
    ranks = list(range(1, len(sorted_counts) + 1))
    plt.plot(ranks, sorted_counts, linestyle='-', color='b', alpha=0.5)
    plt.suptitle(f'Verteilung der Senti-Werte in Relation zum Rang', fontsize=16, fontweight="bold")
    # plt.title(f'Gesamtgröße des Vokabulars: {len(ranks)} Wörter. X-Achse zur Übersichtlichkeit begrenzt')
    plt.xlabel('Rang')
    plt.ylabel('Senti-Wert')
    plt.grid(True)
    fig = plt.gcf()
    fig.savefig(f'data/results/plots/senti/distribution_combination.png')
    plt.close(fig)


def plot_mean_sentiments_as_bar_plot():
    df = pd.read_csv('data/results/mean_senti.csv')
    df = df.sort_values(by=['rank'], ascending=False)
    words = df['word'].tolist()
    ranks = df['rank'].tolist()
    words_with_ranks = [f'{ranks[i]}. {words[i]}' for i in range(len(words))]
    sentis = df['mean_senti'].tolist()
    # creating the bar plot
    plt.barh(words_with_ranks, sentis, color='maroon',
            height=0.5)
    for i in range(len(words_with_ranks)):
        plt.text(sentis[i], i, f'  {math.ceil(sentis[i] * 1000) / 1000}', va='center', color=f'{"black" if ranks[i] < 4 else "white"}')
    plt.axvline(0, color='black', linewidth=2)
    plt.ylabel("Schlagwörter")
    plt.xlabel("Durchschnittlicher WEAT-Wert, politisch + sentiment")
    plt.title("Durchschnittliche Wertungen der Schlagwörter über alle Epochen, \n1949-2023")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    fig.savefig('data/results/plots/senti/mean_senti.png')
    plt.close(fig)


# TODO: delete
def plot_mean_senti_over_epochs_as_bar_plot():
    results = experiment.calculate_mean_sentiment_over_all_keywords()
    senti = [results[i].get('mean_senti') for i in range(len(results))]
    epochs = [results[i].get('epoch') for i in range(len(results))]
    # creating the bar plot
    plt.bar(epochs, senti, color='maroon')
    plt.ylabel("durchschnittliche Wertung der Schlagwörter")
    plt.xlabel("Epochen")
    plt.title("Durchschnittliche Wertungen aller Schlagwörter in den jeweiligen Epochen, \n1949-2023")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    fig.savefig('data/results/plots/senti/mean_senti_for_epochs.png')
    plt.close(fig)


def plot_senti_minima_for_epochs_as_bar_plot():
    minima = experiment.get_senti_minima_for_epochs()
    # creating the bar plot
    plt.bar([entry['name'] for entry in minima.values()], [entry['count'] for entry in minima.values()], color='maroon')
    plt.ylabel("Anzahl der Wertungs-Minima")
    plt.xlabel("Epochen")
    plt.title("Wertungs-Minima aller Schlagwörter in den jeweiligen Epochen, \n1949-2023")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    fig.savefig('data/results/plots/senti/senti_minima.png')
    plt.close(fig)


# Word Associations
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


# Code copied from https://github.com/ezosa/Diachronic-Embeddings/blob/master/embeddings_drift_tsne.py
# maybe not much more useful than lists of nearest neighbors :/
def plot_words_from_time_epochs_tsne(epochs, target_word, aligned_base_folder, k=15, perplexity=30, mode_gensim=True, keep_doubles=False, iter=1000):
    # plot target word across all timeslices
    print("\nPlotting target word...")
    print("Target word: ", target_word)
    target_vectors = experiment.prepare_target_vectors_for_tsne(epochs, target_word, aligned_base_folder, mode_gensim, k, keep_doubles)
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
        # ensure correct order of epochs
        unique_types = ['target_word'] + [item for item in sorted(df['type'].unique()) if item != 'target_word']
        df.sort_values(by=['type', 'words'],
                       key=lambda x:  x.map(
                           {t: i for i, t in enumerate(unique_types)}), inplace=True)
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
        fig.savefig(f'data/results/plots/word_associations/tsne/tsne_{target_word}_{"_".join(map(str, epochs))}_perpl{perplexity}_k{k}_{"gensim" if mode_gensim else "sklearn"}{"_doubles" if keep_doubles else ""}_steps_{iter}.png')
        plt.close()


def plot_tsne_according_to_occurrences(words='all', k=15, perplexity=30, mode_gensim=True, keep_doubles=True, iter=1000):
    # iterate rows in keywords_merged.csv
    df = pd.read_csv('data/keywords_merged.csv')
    for index, row in df.iterrows():
        # if word never occurs, ignore
        if row.first_occ_epoch != 0 and (words == 'all' or row.keyword in words):
            # check if resp. start_epoch-folder exists
            aligned_base_folder = f'data/models/aligned_models/start_epoch_{row.first_occ_epoch}{f"_lh_{row.loophole}" if not str(0) in row.loophole else ""}'
            if os.path.isdir(aligned_base_folder):
                necessary_epochs = [item for item in range(row.first_occ_epoch, row.last_occ_epoch + 1) if str(item) not in row.loophole]
                # plot words with more than one epoch only
                if len(necessary_epochs) > 1:
                    if k == 'flex':
                        if len(necessary_epochs) <= 3:
                            final_k = 18
                        elif 6 > len(necessary_epochs) > 3:
                            final_k = 12
                        else:
                            final_k = 8
                    else:
                        final_k = k
                    plot_words_from_time_epochs_tsne(necessary_epochs, row.keyword, aligned_base_folder, final_k, perplexity, mode_gensim, keep_doubles, iter)
            else:
                print(f"ERROR! Folder {aligned_base_folder} does not exist. "
                      f"Please re-do alignment and/or check your folder structure!")


# TODO: delete
def plot_nearest_neighbors_as_word_clouds():
    df = pd.read_csv('data/results/aggregated_nearest_neighbors.csv')
    keywords = df['Keyword'].tolist()
    for kw in keywords:
        row = df[df['Keyword'] == kw].iloc[0]
        word_sim_dict = {}
        word_sim_dict.update({row['Keyword']: 4.5})
        word_sim_dict.update({row[f'Word_{i}']: row[f'Similarity_{i}'] for i in range(1, 21) if not pd.isna(row[f'Word_{i}'])})

        # Create a WordCloud object with font_size parameter
        wordcloud = WordCloud(width=800, height=400, background_color='white', random_state=42)

        # Generate the word cloud using the frequencies
        wordcloud.generate_from_frequencies(word_sim_dict)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        fig = plt.gcf()
        fig.savefig(f'data/results/plots/word_associations/word_cloud_{kw}.png')
        plt.close(fig)


# TODO: delete
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
            path = f'data/results/plots/word_associations/dist_development_{kw}_plot.png'
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


def plot_cosine_developments_of_word_groups(show_result_groups=True):
    # retrieve keywords that should be compared
    df = pd.read_csv('data/keyword_comparing.csv')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords_merged.csv')
    # iterate these compare groups
    for index, row in df.iterrows():
        # from main word, calculate the similarity for each other word for each epoch
        main_word = row.main_word
        other_words = [row[column] for column in ['second_word', 'third_word', 'fourth_word', 'fifth_word'] if pd.notna(row[column])]
        # get necessary epochs
        kw_row = keywords_df[keywords_df['keyword'] == main_word].iloc[0]
        necessary_epochs = [item for item in range(kw_row.first_occ_epoch, kw_row.last_occ_epoch + 1) if
                            str(item) not in kw_row.loophole]
        results = experiment.calculate_cosine_similarity_between_word_group(main_word, other_words, necessary_epochs)
        # plot as many lines as other words
        title = f'Entwicklung im Verhältnis zum Schlagwort {main_word}'
        path = f'data/results/plots/word_associations/comparing/comparing_development_{main_word}_{"_".join(other_words)}_plot.png'
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
            # add labels to points
            for a, txt in enumerate(results[i]):
                if results[i][a] is not None:
                    plt.text(x[a], results[i][a], f'{results[i][a]:.2f}', color=colors[i], ha='center', va='bottom')
        max_val = max([max(value for value in sublist if value is not None) for sublist in results if sublist != [None]])
        if show_result_groups:
            # plot blue areas to give a hint on the dimensions
            blue_colors = plt.cm.Blues(np.linspace(0.2, 0.9, 5))
            labels = ['wenig/gar nicht ähnlich', 'geringfügig ähnlich', 'moderat ähnlich', 'ähnlich', 'sehr ähnlich']
            for index in range(1, 6):
                max_slice = index * 0.2
                min_slice = (index - 1) * 0.2 if index != 1 else -0.2
                mean_slice = np.mean([max_slice, min_slice]) if index != 1 else 0.1
                ax.axhspan(min_slice, max_slice, facecolor=blue_colors[index - 1], alpha=0.35)
                # if the slice contains result points, plot its' name to the side
                if not max_val <= min_slice:
                    plt.text(0.1, mean_slice,
                             labels[index - 1],
                             color=blue_colors[index - 1], fontsize=10)
        ax.set_ylim([-0.025, max_val + 0.025])
        # plot legend
        plt.legend()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print(f"plot saved")


def plot_cosine_developments_nearest_neighbors_heatmap():
    # retrieve keywords that should be compared
    agg_df = pd.read_csv('data/results/aggregated_nearest_neighbors.csv')
    df = pd.read_csv('data/results/nearest_neighbors.csv')
    epoch_df = pd.read_csv('data/epochs.csv')
    keywords = agg_df['Keyword'].tolist()
    for kw in keywords:
        # retrieve list of similar words
        kw_df = df[df['Keyword'] == kw]
        word_keys = [f'Word_{num}' for num in range(1, 21)]
        all_words = agg_df[agg_df['Keyword'] == kw][word_keys].iloc[0].dropna().tolist()
        sim_keys = [f'Similarity_{num}' for num in range(1, 11)]
        all_similarities = []
        epochs = []
        for index, row in kw_df.iterrows():
            words = row[word_keys[:10]].tolist()
            similarities = row[sim_keys].tolist()
            # retrieve their similarities in the different epochs. if not given, say 0
            all_epoch_sims = [similarities[words.index(word)] if word in words else 0 for word in all_words]
            epochs.append(row.Epoch)
            all_similarities.append(all_epoch_sims)
        # plot
        heatmap_data = np.array(all_similarities).transpose()
        plt.imshow(heatmap_data, cmap='OrRd', aspect='auto')
        plt.colorbar(label='Cosine Similarity')
        written_epochs = [epoch_df[epoch_df['epoch_id'] == epoch]['written_form_short'].iloc[0] for epoch in epochs]
        # plot grid
        for i in range(len(heatmap_data) + 1):
            plt.axhline(i - 0.5, color='grey', linewidth=0.8)
        for i in range(len(heatmap_data[1]) + 1):
            plt.axvline(i - 0.5, color='grey', linewidth=0.8)
        plt.xticks(np.arange(len(written_epochs)), written_epochs, rotation=90)
        plt.yticks(np.arange(len(all_words)), all_words)
        plt.xlabel("Epochen")
        plt.ylabel("Wörter")
        plt.title(f"Word Similarities Heatmap: {kw}")
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(f'data/results/plots/word_associations/heatmaps/{kw}_heatmap.png')
        plt.close(fig)
        print(f"plot for kw {kw} saved")


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
