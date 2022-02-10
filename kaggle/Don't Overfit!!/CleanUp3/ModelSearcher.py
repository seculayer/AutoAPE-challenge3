from CleanUp3.Model.ModelOrigin import ModelOrigin
from CleanUp3.Common.LoadData import LoadData
from CleanUp3.Model.CVFactory import CVFactory



if __name__ == '__main__':
    DATA_DICT = LoadData().get_data()
    model_nm_list = ['LASSO','LOGISTIC','EXTRATREE','ABC','RANDOMFOREST','SGD','SVC']
    CV = CVFactory().create('GRIDCV')
    for model_nm in model_nm_list:
        model, parameter_grid = ModelOrigin().create(model_nm)
        prediction,_ = CV.Learn(model = model,DATA_DICT=DATA_DICT,parameter_grid=parameter_grid)

    # Pred2Csv(prediction,'kaggle_hard_hard.csv')
