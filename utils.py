import pickle
import csv
import pandas as pd


def make_pickle(filepath, data):
    """
    take data and store it at the given file path (with pickle)
    """
    with open(filepath, "wb") as fp:  # Pickling
        pickle.dump(data, fp)


def unpickle(filepath):
    """
    get stored data at given filepath (saved with pickle)
    """
    with open(filepath, "rb") as fp:  # Unpickling
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


def load_keywords():
    """
    helper method to quickly load keywords for the experiment
    :return: all keywords from keywords_merged.csv as a list
    """
    df = pd.read_csv('data/keywords_merged.csv')
    return df['keyword'].tolist()


def get_epoch_written_form_short(epoch_id):
    """
    fetch short written form of epoch
    :param epoch_id: number signifying an historical epoch defined in epochs.csv
    :return: short written form for epoch, e.g. '1950er'
    """
    epochs_df = pd.read_csv('data/epochs.csv')
    return epochs_df[epochs_df['epoch_id'] == epoch_id]['written_form_short'].iloc[0]
