import os

import numpy as np
from sklearn.model_selection import train_test_split

from Configuration import Configuration
from common.controller.PreProcess import PreProcess
from common.model.ClassFile import ClassFile
from common.model.Stats import Stats
from commonsLib import loggerElk
from service.ClassifyFactory import ClassifyFactory, get_category


class TextClass:

    def __init__(self, conf: Configuration):
        self.conf = conf
        self.logger = loggerElk(__name__, True)

    def _initialize(self, nlp=None, dictionary=None):
        self.logger.Information('TextClass - loading configuration...')
        if dictionary is not None:
            self.conf.dictionary = dictionary
        if nlp is not None:
            self.logger.Information('TextClass - pre-processing...')
            self.pre_process = PreProcess(self.conf, nlp=nlp)
        self.classify = ClassifyFactory(self.conf)

    def train_by(self, model, data, nlp, dictionary):
        self._initialize(nlp=nlp, dictionary=dictionary)

        stats = Stats()
        training_set = os.path.join(self.conf.working_path, data)
        v_list = ClassFile.list_files_ext(training_set, ".gram")
        X = list()
        y = list()
        corpus_size = len(v_list)
        vector_size = 0
        for f in v_list:
            vector = self.pre_process.get_tfidf_from_vectorizer(ClassFile.file_to_list(f))
            if vector is not None:
                category = get_category(f)
                n_vector = vector.toarray()[0]
                vector_size = len(n_vector)
                X.append(n_vector)
                y.append(category)

        stats.classifier = model
        try:
            X = np.array(X).reshape((corpus_size, vector_size)).astype(np.float16)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

            response = []
            if nlp:
                # Basic classifiers
                if model == ClassifyFactory.RANDOM_FOREST:
                    self.logger.Information(f'TextClass - training {model}...')
                    response = self.classify.launch_random_forest(X_train, y_train, X_test, True)
                    self.classify.show_metrics(y_test, response, stats=stats)

                else:
                    response = []
                    stats.info = 'Unsuitable training model. ' \
                                 'Should be one of: RANDOM_FOREST'
                    stats.result = 'WRONG_MODEL'

            stats.update_response(response)
        except:
            self.logger.Error(f"TextClass - {model} training... allocation memory error!")

        return stats

    def predict_by(self, model, nlp, data=None, file_data=None, file_url=None):
        self._initialize(nlp=nlp)

        stats = Stats()
        response = []
        if nlp:
            vector_list = list()
            if data is None:
                if file_data:
                    vector_list = self.pre_process.transform_text(text=file_data)  # just one document
                elif file_url:
                    vector_list = self.pre_process.transform(file=file_url)  # just one url/document
            else:
                vector_list = self.pre_process.transform_data(data=data)

            stats.classifier = model

            X_test = list()
            for vector in vector_list:
                X_test.append(vector.toarray()[0])
            X_test = np.array(X_test).reshape((len(vector_list), len(X_test[0])))

            # Basic classifiers
            if model == ClassifyFactory.RANDOM_FOREST:
                self.logger.Information(f'TextClass - {model} prediction...')
                response = self.classify.launch_random_forest(None, None, X_test, False)
                self.classify.show_metrics(None, response, stats=stats)

            else:
                response = []
                stats.info = 'Unsuitable prediction model. ' \
                             'Should be one of: RANDOM_FOREST)'
                stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    @staticmethod
    def check_model(model=None):
        # is a available classification method
        if not (model == ClassifyFactory.RANDOM_FOREST):
            return False

        return True

    def check_source(self, data=None):
        # has right source type
        if data is None:
            if not ClassFile.has_text_file(self.conf.working_path):
                return False
        elif not ClassFile.has_text_file(os.path.join(self.conf.working_path, data)):
            return False

        return True

    def check_encoder(self):
        if not ClassFile.list_files_ext(self.conf.working_path, self.conf.vectorizer_type):
            return False

        return True
