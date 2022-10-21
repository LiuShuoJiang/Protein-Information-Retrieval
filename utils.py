import pandas as pd
import numpy as np


def fetch_file_names(file_path):
    data_file = pd.read_csv(file_path, index_col=None, header=None)
    path_list = data_file[0].tolist()
    lines = data_file[1].astype(np.int32).tolist()
    return path_list, lines
