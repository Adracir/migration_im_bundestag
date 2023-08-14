import pickle


def unpickle(file_name):
    with open(file_name, "rb") as fp:  # Unpickling
        unpickled = pickle.load(fp)
    return unpickled