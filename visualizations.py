import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils
from gensim.models import KeyedVectors


def plot_frequencies():
    # TODO: maybe also plot expected frequencies to compare
    freqs_df = pd.read_csv('data/results/freqs.csv', encoding='cp1252')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords.csv')
    # keywords = utils.load_keywords()
    keywords = keywords_df['word'].tolist()
    combine_groups = keywords_df['combine_group'].tolist()
    epochs = epochs_df['epoch_id'].tolist()
    # TODO: Obacht! freqs and keywords and epochs should be at same status for this to work!
    #  Maybe implement cleverer way
    indices_done = []
    for i in range(0, len(keywords)):
        combine_group = combine_groups[i]
        # case 1: word has already been plotted
        if i in indices_done:
            continue
        # case 2: word does not need to be combined with other spelling and has not been plotted yet
        if combine_group == 0:
            kw = keywords[i]
            kw_freqs_df = freqs_df[freqs_df['keyword'] == kw]
            freqs = kw_freqs_df['freq'].tolist()
            title = f'Häufigkeiten des Schlagwortes {kw}'
            path = f'data/results/plots/freq_{kw}_plot.png'
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
            indices_done.extend(combine_group_indices)
            title = f"Häufigkeiten der Schlagwörter {', '.join(combined_kws)}"
            path = f"data/results/plots/freq_combined_{'_'.join(combined_kws)}_plot.png"
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
        plt.plot(x, freqs, 'r-')
        # set tight layout (so that nothing is cut out)
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print("plot saved")


def plot_comparing_frequencies():
    freqs_df = pd.read_csv('data/results/freqs.csv', encoding='cp1252')
    epochs_df = pd.read_csv('data/epochs.csv')
    keywords_df = pd.read_csv('data/keywords.csv')
    # keywords = utils.load_keywords()
    keywords = keywords_df['word'].tolist()
    compare_groups = keywords_df['compare_group'].tolist()
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
        path = f"data/results/plots/freq_compared_{'_'.join(compared_kws)}_plot.png"
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
        colors = ['r', 'b', 'g', 'o']
        for a in range(0, len(freqs)):
            plt.plot(x, freqs[a], f'{colors[a]}-', label=compared_kws[a])
        # show legend
        plt.legend()
        # set tight layout (so that nothing is cut out)#
        plt.tight_layout()
        # save diagram
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(path)
        plt.close(fig)
        print("plot saved")


# Code copied from https://github.com/ezosa/Diachronic-Embeddings/blob/master/embeddings_drift_tsne.py
from sklearn.manifold import TSNE  # needed to be installed
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # needed to be installed
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from matplotlib import collections as mc


# label points with words
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if i % 2 == 0:
            ax.text(point['x'] + .02, point['y'], str(point['val']))
        else:
            ax.text(point['x'] + .02, point['y'] - .02, str(point['val']))


# maybe not much more useful than lists of nearest neighbors :/
def plot_words_from_time_epochs_tsne(epochs, target_words, k=10, perplexity=20, mode_gensim=True, keep_doubles=True):
    # plot each target word across all timeslices
    print("target words: ", len(target_words))
    print(target_words)
    print("\nPlotting each target word...")
    for target in target_words:
        print("Target word: ", target)
        target_vectors = {}
        for epoch in epochs:
            model_wv_path = f'data/models/epoch{epoch}_lemma_200d_7w_cbow.wordvectors'
            # print("Year: ", model_years[year_index])
            word_vectors = KeyedVectors.load(model_wv_path)
            vocab = list(word_vectors.index_to_key)
            target_word_year = f'{target}_{epoch}'
            target_vectors[target_word_year] = {}
            target_vectors[target_word_year]['vector'] = word_vectors[target]
            target_vectors[target_word_year]['type'] = 'target_word'
            if mode_gensim:
                word_sim = word_vectors.most_similar(positive=target, topn=k)
            else:
                target_word_vec = [word_vectors[target]]
                vocab_sim = [cosine_similarity(target_word_vec, [word_vectors[vocab_word]]) for vocab_word in vocab if
                             vocab_word != target]
                word_sim = [(w, s) for s, w in sorted(zip(vocab_sim, vocab), reverse=True)][:k]
            for ws in word_sim:
                if keep_doubles:
                    ws_key = f'{ws[0]}_{epoch}'
                    target_vectors[ws_key] = {}
                    target_vectors[ws_key]['vector'] = word_vectors[ws[0]]
                    target_vectors[ws_key]['type'] = (epoch)
                    target_vectors[ws_key]['sim'] = ws[1]
                elif (ws[0] not in target_vectors.keys()) or (ws[0] in target_vectors.keys()
                                                            and ws[1] > target_vectors[ws[0]]['sim']):
                    # throwing away double existing words makes them invisible in future epochs
                    target_vectors[ws[0]] = {}
                    target_vectors[ws[0]]['vector'] = word_vectors[ws[0]]
                    target_vectors[ws[0]]['type'] = (epoch)
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
            tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000, learning_rate=100.0)
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
                row1 = df_target.iloc[row_num,:]
                row2 = df_target.iloc[row_num+1,:]
                p1 = (row1['tsne-one'], row1['tsne-two'])
                p2 = (row2['tsne-one'], row2['tsne-two'])
                lines.append([p1, p2])
            lc = mc.LineCollection(lines, linewidths=1)
            ax.add_collection(lc)
            fig = ax.get_figure()
            fig.savefig(f'data/results/plots/tsne_{target}_{"_".join(map(str, epochs))}_perpl{perplexity}_k{k}_{"gensim" if mode_gensim else "sklearn"}{"_doubles" if keep_doubles else ""}.png')
            plt.close()


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


plot_words_from_time_epochs_tsne([1, 2, 3, 4, 5, 6, 7, 8], target_words=['Flüchtling'],
                                 perplexity=30, k=10, keep_doubles=False)
