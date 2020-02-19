import os
import pickle
import sys

import numpy as np

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

    def do_train(self, x, y):
        self.clf.fit(x, y)

    @staticmethod
    def _get_indexes_max_value(ln):
        max_value = max(ln)
        if ln.count(max_value) > 1:
            return [i for i, x in enumerate(ln) if x == max(ln)]
        else:
            return ln.index(max(ln))

    def get_prediction(self, x):
        response = list()
        try:
            predicted = list()
            classes, probabilities = self.do_predict_prob(x)
            probability_list = probabilities.astype(np.float16).tolist()
            for prob in probability_list:
                soft_max = self._get_indexes_max_value(prob)
                predicted.append(str(classes[soft_max]))

            response.append(predicted)
            response.append(classes)
            response.append(probability_list)
        except Exception as e:
            self.logger.Debug('IClassify::get_prediction::{}'.format(e), sys.exc_info())
            return None

        return response

    def do_predict(self, x):
        if self.clf is None:
            self.logger.Information("Please, load model")
            return None

        y = self.clf.predict(x)
        return y

    def do_predict_prob(self, x):
        if self.clf is None:
            self.logger.Information("Please, load model", sys.exc_info())
            return None

        y = self.clf.predict_proba(x)
        return self.clf.classes_, y

    def load_model(self, model_name, s3_service=None):
        try:
            self.logger.Information('IClassify::loading model...')
            if s3_service:
                model_bytes = s3_service.get_byte_file(s3_service.domain + "/" + model_name)
                self.clf = pickle.loads(model_bytes)
            else:
                path = os.path.join(self.conf.working_path, model_name)
                self.clf = ClassFile.load_model(path)
        except Exception as e:
            self.clf = None
            self.logger.Debug('IClassify::LOAD::{} - model -> {}'.format(e, model_name), sys.exc_info())

    def save_model_s3(self, model_name, s3_service):
        try:
            if self.clf:
                self.logger.Information('IClassify::saving model...')
                pickle_byte_obj = pickle.dumps(self.clf)
                path = s3_service.domain + "/" + model_name
                s3_service.upload_file(path, pickle_byte_obj)
        except Exception as e:
            self.clf = None
            self.logger.Debug('IClassify::SAVE-S3::{} - model -> {}'.format(e, model_name), sys.exc_info())

    def save_model(self, model_name):
        try:
            if self.clf:
                self.logger.Information('IClassify::saving model...')
                path = os.path.join(self.conf.working_path, model_name)
                ClassFile.save_model(path, self.clf)
        except Exception as e:
            self.clf = None
            self.logger.Debug('IClassify::SAVE::{} - model -> {}'.format(e, model_name), sys.exc_info())

    def has_model(self):
        return self.clf and self.clf is not None
