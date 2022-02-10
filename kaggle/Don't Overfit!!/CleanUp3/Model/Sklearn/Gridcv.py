from CleanUp3.Util.Feature_Select import Feature_Select
from sklearn.metrics import roc_auc_score
from CleanUp3.Model.HPOAbstract import HPOAbstract
import numpy as np
import pandas as pd
class Gridcv(HPOAbstract):

    def Learn(self,model, DATA_DICT,parameter_grid):
        _, X_train_selected = self.train_model(model, DATA_DICT)
        grid_search = self.greidsh(model, DATA_DICT,parameter_grid, X_train_selected)
        prediction, _ = self.train_model(model=grid_search.best_estimator_,DATA_DICT=DATA_DICT,folds=self.repeated_folds )
        return prediction, _

    def train_model(self, model, DATA_DICT, parameter_grid=None, folds=None):
        return super().train_model(model, DATA_DICT, parameter_grid, folds,
                                   _ready_func=self._ready_func,
                                   _train_func = self._train_func,
                                   _result_func=self._result_func)

    def _ready_func(self, folds):
        if folds is None:
            folds = self.folds
        scores_train = []
        scores_valid = []
        return folds, scores_train, scores_valid

    def _train_func(self, model, DATA_DICT, fold_data_dict,scores_train,scores_valid, prediction, counter):
        y_pred_train = model.predict(fold_data_dict['X_train_fold']).reshape(-1,1)
        train_score = roc_auc_score(fold_data_dict['y_train_fold'],y_pred_train)
        scores_train.append(train_score)

        y_pred_valid = model.predict(fold_data_dict['X_valid_fold']).reshape(-1,1)
        valid_score = roc_auc_score(fold_data_dict['y_valid_fold'],y_pred_valid)
        scores_valid.append(valid_score)

        # 예측 확률
        try:
            predict = model.predict(DATA_DICT['X_test'])
            # y_pred = model.predict_proba(DATA_DICT['X_test'])[:,1]
        except AttributeError:
            predict = model.score(fold_data_dict['X_valid_fold'],fold_data_dict['y_valid_fold'])#[:,1]
        prediction = pd.concat([prediction, pd.DataFrame(predict)], axis=1)
        counter += 1
        return scores_train,scores_valid,prediction,counter

    def _result_func(self, prediction, DATA_DICT, model,scores_train,scores_valid):
        # print('n_split & counter')
        # print(folds.get_n_splits())
        # print(counter)
        prediction=prediction.mean(axis=1)
        # prediction /= counter
        # prediction /= folds.get_n_splits()
        # print(prediction)

        X_train_selected = DATA_DICT['X_train'][Feature_Select(model)]
        print(f'평균 학습 정확도: {np.mean(scores_train):.4f}, 표준편차: {np.std(scores_train):.4f}.')
        print(f'평균 평가 정확도: {np.mean(scores_valid):.4f}, 표준편차: {np.std(scores_valid):.4f}.')
        return prediction, X_train_selected