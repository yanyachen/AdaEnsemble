import pickle


def count_lines(file_name):
    with open(file_name, "rb") as file:
        count = sum(1 for _ in file)
    return count


def save_pickle(obj, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    return None


def load_pickle(file_name):
    with open(file_name, "rb") as file:
        obj = pickle.load(file)
    return obj
