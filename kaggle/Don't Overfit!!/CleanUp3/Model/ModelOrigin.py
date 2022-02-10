from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from CleanUp3.Common.Constant import Constant
from sklearn.linear_model import Lasso
class ModelOrigin():
    @staticmethod
    def create(model_nm):
        print(f'--------------------{model_nm}-----------------------')
        if model_nm =='LOGISTIC':
            model = LogisticRegression
        elif model_nm =='RANDOMFOREST':
            model = RandomForestClassifier
        elif model_nm =='EXTRATREE':
            model = ExtraTreesClassifier
        elif model_nm =='ABC':
            model = AdaBoostClassifier
        elif model_nm =='SGD':
            model = SGDClassifier
        elif model_nm =='SVC':
            model = SVC
        elif model_nm =='LASSO':
            model = Lasso(alpha=0.02)
        else:
            print('abstract error')
            raise
        parameter_grid = Constant()._get_param(model_nm=model_nm)

        return model, parameter_grid