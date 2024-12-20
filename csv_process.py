import pandas as pd


def csv_process(file_path):
    df = pd.read_csv(file_path)
    return df.head(10)