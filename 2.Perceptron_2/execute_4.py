import os 
import pandas as pd

if __name__ == "__main__":
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    df = pd.read_csv(DATASET_URL, header=None, encoding='utf-8')
    
    print(df.tail())