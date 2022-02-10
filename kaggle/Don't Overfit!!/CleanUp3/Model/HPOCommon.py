from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from CleanUp3.Model.ModelOrigin import ModelOrigin
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
class HPOCommon:

    def greidsh(self,model, DATA_DICT, parameter_grid,X_train_selected):
        try:
            model = model()
        except:
            pass
        grid_search = GridSearchCV(model, param_grid=parameter_grid, cv=self.folds, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_selected, DATA_DICT['y_train'])
        print(f'죄고점수: {grid_search.best_score_}')
        print(f'최종 파라미터: {grid_search.best_params_}')
        return  grid_search

    def rfecv(self, model ,min_features_to_select=12, step=15):
        # robust_roc_auc = make_scorer(self.scoring_roc_auc)
        # model, self.parameter_grid = ModelOrigin().create(model_nm)
        feature_selector = RFECV(model, min_features_to_select=min_features_to_select, step=step,
                                 verbose=0, cv=self.repeated_folds, n_jobs=-1) #, scoring=robust_roc_auc
        return feature_selector

    def Robust(self,DATA_DICT):
        columns = DATA_DICT['X_train'].columns
        DATA = RobustScaler().fit_transform(np.concatenate((DATA_DICT['X_train'], DATA_DICT['X_test']), axis=0))
        DATA_DICT['X_train'] = pd.DataFrame(DATA[:250])
        DATA_DICT['X_train'].columns = columns
        DATA_DICT['X_test'] = pd.DataFrame(DATA[250:])
        DATA_DICT['X_test'].columns = columns
        return DATA_DICT

    def rfecv_greidsh(self,model, parameter_grid,X_train_selected,y_train_fold):
        grid_search = GridSearchCV(model.estimator_, param_grid=parameter_grid, cv=self.repeated_folds, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_selected, y_train_fold)

        print(f'죄고점수: {grid_search.best_score_}')
        print(f'최종 파라미터: {grid_search.best_params_}')

        return  grid_search
if __name__ == '__main__':
    modelabstract = HPOCommon('LOGISTIC')
    scores_logreg, prediction = modelabstract.GetBestParm()
    # modelabstract.train_model()
    # grid_search = modelabstract.greidsh()
    # scores_logreg, prediction = modelabstract.train_model(folds=1, model=grid_search.best_estimator_)