import pickle
import csv
from gensim.models import KeyedVectors
from gensim import matutils
from numpy import dot
import pandas as pd


def make_pickle(file_name, data):
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(data, fp)


def unpickle(file_name):
    with open(file_name, "rb") as fp:  # Unpickling
        unpickled = pickle.load(fp)
    return unpickled


def write_info_to_csv(output_file_path, arr, mode='w', encoding='utf-8'):
    """
    write array to csv
    :param output_file_path: path to which the file should be saved
    :param arr: containing all row values that should be written
    :param mode: csv writer mode: 'w' for writing to a new file, 'a' for appending an existing one
    :param encoding: Encoding of the resulting file
    """
    with open(output_file_path, mode=mode, newline='', encoding=encoding) as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(arr)


def load_model(name):
    return KeyedVectors.load(f"data/models/{name}")


def load_keywords():
    df = pd.read_csv('data/keywords.csv')
    return df['word'].tolist()


def similarity(wv1, wv2):
    return dot(matutils.unitvec(wv1), matutils.unitvec(wv2))


def get_epoch_written_form_short(epoch_id):
    epochs_df = pd.read_csv('data/epochs.csv')
    epochs = epochs_df['epoch_id'].tolist()
    written_forms = epochs_df['written_form_short'].tolist()
    i = epochs.index(epoch_id)
    return written_forms[i]
