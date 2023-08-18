import pickle
import csv


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
