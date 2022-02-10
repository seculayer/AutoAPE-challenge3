from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score
from CleanUp3.Model.HPOAbstract import HPOAbstract

import pandas as pd

class Rfecv(HPOAbstract):

    def Learn(self,model, DATA_DICT,parameter_grid):
        rfecv_model = self.rfecv(model)
        prediction = self.train_model(model=rfecv_model,DATA_DICT=DATA_DICT,parameter_grid = parameter_grid)
        return prediction

    def train_model(self,model, DATA_DICT, parameter_grid=None, folds=None):
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

    def _train_func(self, model, DATA_DICT, fold_data_dict,parameter_grid, prediction, counter):
        X_train_selected = model.transform(fold_data_dict['X_train_fold'])
        val_X_important_features = model.transform(fold_data_dict['X_valid_fold'])
        test_important_features = model.transform(DATA_DICT['X_test'])
        grid_search = self.rfecv_greidsh(model, parameter_grid, X_train_selected,fold_data_dict['y_train_fold'])

        val_y_pred = grid_search.best_estimator_.predict(val_X_important_features)
        val_mse = mean_squared_error(fold_data_dict['y_valid_fold'], val_y_pred)
        val_mae = mean_absolute_error(fold_data_dict['y_valid_fold'], val_y_pred)
        val_roc = roc_auc_score(fold_data_dict['y_valid_fold'], val_y_pred)
        val_r2  = r2_score(fold_data_dict['y_valid_fold'], val_y_pred)

        if val_mse < 0.18:
            message = '<OK'
            predict = grid_search.best_estimator_.predict(test_important_features)
            prediction = pd.concat([prediction, pd.DataFrame(predict)], axis=1)
        else:
            message = 'skipping'
        print("{0:2}      | {1:.4f}   |  {2:.4f}   |  {3:.4f}    |  {4:.4f}    |  {5:3}   |   {6}  "
              .format(counter, val_mse, val_mae, val_roc, val_r2, model.n_features_, message))
        counter += 1
        return prediction, counter

    def _result_func(self,prediction):

        prediction=prediction.mean(axis=1)

        return prediction
