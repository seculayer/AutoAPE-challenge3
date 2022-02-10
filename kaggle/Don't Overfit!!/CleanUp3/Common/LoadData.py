
from .Constant import Constant
import pandas as pd
class LoadData:

    def __init__(self):
        self.DATA_PATH = Constant()._get_dataDir()


    def get_data(self):
        train = pd.read_csv(self.DATA_PATH+'train.csv')
        test = pd.read_csv(self.DATA_PATH+'test.csv')
        mission = pd.read_csv(self.DATA_PATH+'sample_submission.csv')
        X_train = train.drop(['id', 'target'], axis=1)
        y_train = train['target']
        X_test = test.drop(['id'], axis=1)
        DATA_DICT={'X_train':X_train,'y_train':y_train,'X_test':X_test,'mission':mission}
        return DATA_DICT
