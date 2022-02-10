
from CleanUp3.Common.LoadData import LoadData
from CleanUp3.Util.Pred2Csv import Pred2Csv
from CleanUp3.Model.CVFactory import CVFactory
from CleanUp3.Model.ModelOrigin import ModelOrigin


class FinalModel():
    DATA_DICT = LoadData().get_data()

    CV = CVFactory().create('RFECV')
    DATA_DICT = CV.Robust(DATA_DICT)
    model, parameter_grid = ModelOrigin().create('LASSO')
    prediction=CV.Learn(model, DATA_DICT,parameter_grid)
    # rfecv_model = CV.rfecv(model)
    # prediction = CV.train_model(model=rfecv_model,DATA_DICT=DATA_DICT,parameter_grid = parameter_grid)
    Pred2Csv(prediction,'kaggle_hard_hard.csv')

if __name__ == '__main__':
    FinalModel()


