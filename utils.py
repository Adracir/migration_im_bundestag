import pickle
import csv
from gensim.models import KeyedVectors
import pandas as pd


def unpickle(file_name):
    with open(file_name, "rb") as fp:  # Unpickling
        unpickled = pickle.load(fp)
    return unpickled


def write_info_to_csv(output_file_path, arr, mode='w'):
    """
    write array to csv
    :param output_file_path: path to which the file should be saved
    :param arr: containing all row values that should be written
    :param mode: csv writer mode: 'w' for writing to a new file, 'a' for appending an existing one
    """
    with open(output_file_path, mode=mode, newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(arr)


def load_model(name):
    return KeyedVectors.load(f"data/models/{name}")


def load_keywords():
    df = pd.read_csv('data/keywords2.csv')
    return df['word'].tolist()
