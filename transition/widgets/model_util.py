import os
import re
from widgets.dugu_util import ToolConfig


class NamedModel:
    def __init__(self):

        if not os.path.exists(ToolConfig.MODEL_PATH):
            os.makedirs(ToolConfig.MODEL_PATH, exist_ok=True)

        self.__config = dict()
        self.__comment = dict()
        self.__name = 'model'

        self.__loaded_path = None
        self.__ckpt = None

    @property
    def loaded_path(self):
        return self.__loaded_path

    @property
    def loaded_ckpt(self):
        return self.__ckpt

    @classmethod
    def load(cls, path, name, ckpt = -1):
        found_file = None
        found_ckpt = -1
        cut = re.compile(r"[$@#]")
        for file in os.listdir(path):
            sss = cut.split(file)
            if len(sss)<4:
                continue
            modelname = sss[0]
            modelckpt = int(sss[3])

            if modelname == name:
                if ckpt == -1 and modelckpt > found_ckpt:
                    found_file, found_ckpt = file, modelckpt
                if ckpt != -1 and modelckpt == ckpt:
                     found_file, found_ckpt = file, modelckpt

        model_util = NamedModel()
        sss = cut.split(found_file)
        model_util.__name = sss[0]
        model_util.__ckpt = found_ckpt
        model_util.__loaded_path = path + found_file
        for x in sss[1].split('&'):
            model_util.__config[x.split('=')[0]] = x.split('=')[1]
        for x in sss[2].split('#'):
            model_util.__comment[x.split('=')[0]] = x.split('=')[1]
        return model_util


    def set_name(self, name):
        self.__name = name
        return self

    def set_config(self, k, v):
        self.__config[k] = v
        return self

    def get_config(self, k):
        return self.__config[k]

    def set_comment(self, k, v):
        self.__comment[k] = v
        return self

    def get_comment(self, k):
        return self.__comment[k]

    def ckpt(self, ckpt_id):
        return self.get_name() + '@' + str(ckpt_id)

    def get_name(self):
        return self.__name + \
               '$' + \
               '&'.join(list(sorted(map(lambda item: item[0] + '=' + str(item[1]), self.__config.items())))) + \
               '#' + \
               '&'.join(list(sorted(map(lambda item: item[0] + '=' + str(item[1]), self.__comment.items()))))



