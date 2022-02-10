from CleanUp3.Model.HPOCommon import HPOCommon
from CleanUp3.Common.Constant import Constant
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import pandas as pd

class HPOAbstract(HPOCommon):
    def __init__(self):
        self.repeated_folds = RepeatedStratifiedKFold(n_splits=20, n_repeats=5, random_state=42)
        self.folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
        self.Constant = Constant()


    def train_model(self,model, DATA_DICT, parameter_grid=None, folds=None,_ready_func=None,_train_func = None,_result_func=None):
        try:
            model = model()
        except:
            pass

        prediction = pd.DataFrame()
        counter = 0
        if str(model)[:5] == 'RFECV':
            print("counter |    mse     |    mae     |    roc    |     r2    | feature_count ")
            print("-------------------------------------------------------------------------------------------------")
        else:
            folds, scores_train, scores_valid = _ready_func(folds)

        for train_idx, valid_idx in self.folds.split(DATA_DICT['X_train'],DATA_DICT['y_train']):
            X_train_fold, X_valid_fold = DATA_DICT['X_train'].iloc[train_idx], DATA_DICT['X_train'].iloc[valid_idx]
            y_train_fold, y_valid_fold = DATA_DICT['y_train'].iloc[train_idx], DATA_DICT['y_train'].iloc[valid_idx]
            fold_data_dict={'X_train_fold':X_train_fold, 'X_valid_fold':X_valid_fold,'y_train_fold':y_train_fold, 'y_valid_fold':y_valid_fold}
            model.fit(X_train_fold,y_train_fold)

            if str(model)[:5] == 'RFECV':
                prediction, counter= _train_func( model, DATA_DICT, fold_data_dict,parameter_grid, prediction, counter)
            else:
                scores_train,scores_valid,prediction,counter = _train_func( model, DATA_DICT, fold_data_dict,scores_train,scores_valid, prediction, counter)

        # 예측 평균
        if str(model)[:5] == 'RFECV':
            prediction = _result_func(prediction)
            return prediction
        else:
            prediction, X_train_selected = _result_func(prediction, DATA_DICT, model,scores_train,scores_valid)
            return prediction, X_train_selected