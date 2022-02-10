import pandas as pd
import numpy as np
def Pred2Csv(data, path):
    print('save data to csv')
    if type(data).__module__ == np.__name__:
        csv = pd.DataFrame(data)
        csv.index += 250
        csv.columns = ['target']
        print(csv)
        csv.to_csv(path, index_label='id', index=True)

    if type(data).__module__ == pd.__name__:
        csv = pd.DataFrame(data.mean(axis=1))
        csv.index += 250
        csv.columns = ['target']
        print(csv)
        csv.to_csv(path, index_label='id', index=True)
