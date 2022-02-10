import configparser

class Configurations(object):
    def __init__(self,conf_path):
        # conf_path = "../conf/default.conf"
        self.config = configparser.ConfigParser()
        self.config.read(conf_path)

    def get(self, section, key):
        return self.config.get(section, key)

if __name__ == '__main__':
    default_config = Configurations()
    print(Configurations().get("DIR_CONFIG", "DIR_DATA"))