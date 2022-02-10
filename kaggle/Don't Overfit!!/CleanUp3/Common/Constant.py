import os
from CleanUp3.Util.Singleton import Singleton
from CleanUp3.Util.Configuration import Configurations

class Constant(metaclass=Singleton):
    def __init__(self):
        self.__FILE_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
        self.CONFIG = Configurations(self.__FILE_REAL_PATH+'/../conf/default.conf')

    def _get_dataDir(self):
        return self.__FILE_REAL_PATH+'/../../'+self.CONFIG.get("DIR_CONFIG", "DIR_DATA")

    def _get_param(self, model_nm : str) -> dict :
        return eval(self.CONFIG.get("FARAMS_CONFIG", model_nm))
if __name__ == '__main__':
    # __FILE_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
    Constant = Constant()
    print(Constant._get_dataDir())
    print(Constant._get_param(model_nm='LASSO'))