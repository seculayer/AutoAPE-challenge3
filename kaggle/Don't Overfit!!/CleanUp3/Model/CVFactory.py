from CleanUp3.Model.Sklearn.Gridcv import Gridcv
from CleanUp3.Model.Sklearn.Rfecv import Rfecv
class CVFactory():
    @staticmethod
    def create(model_nm):
        print(f'--------------------{model_nm}-----------------------')
        if model_nm =='RFECV':
            model = Rfecv()
        elif model_nm =='GRIDCV':
            model = Gridcv()
        else:
            print('abstract error')
            raise
        return model