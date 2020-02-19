import os
import sys

from Configuration import Configuration
from common.model.ClassFile import ClassFile
from commonsLib import loggerElk


class IClassify(object):
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.clf = None
        self.logger = loggerElk(__name__, True)

    def initialize(self, *args):
        pass

    @staticmethod
    def _get_indexes_max_value(ln):
        max_value = max(ln)
        if ln.count(max_value) > 1:
            return [i for i, x in enumerate(ln) if x == max(ln)]
        else:
            return ln.index(max(ln))

    def load_model(self, model_name):
        try:
            self.logger.Information('IClassify::loading model...')
            path = os.path.join(self.conf.working_path, model_name)
            self.clf = ClassFile.load_model(path)
        except Exception as e:
            self.clf = None
            self.logger.Debug('IClassify::LOAD::{} - model -> {}'.format(e, model_name), sys.exc_info())

    def save_model(self, model_name):
        try:
            if self.clf:
                self.logger.Information('IClassify::saving model...')
                path = os.path.join(self.conf.working_path, model_name)
                ClassFile.save_model(path, self.clf)
        except Exception as e:
            self.logger.Debug('IClassify::SAVE::{} - model -> {}'.format(e, model_name), sys.exc_info())

    def has_model(self):
        return self.clf and self.clf is not None
