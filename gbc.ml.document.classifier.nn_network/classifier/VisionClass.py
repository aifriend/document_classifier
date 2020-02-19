import os
import sys

from keras_preprocessing.image import ImageDataGenerator

from Configuration import Configuration
from common.model.ClassFile import ClassFile
from common.model.Stats import Stats
from commonsLib import loggerElk
from service.ClassifyFactory import ClassifyFactory


class VisionClass:

    def __init__(self, conf: Configuration):
        self.conf = conf
        self.logger = loggerElk(__name__, True)

    def _initialize(self):
        self.logger.Information('VisionClass - loading configuration...')
        self.classify = ClassifyFactory(self.conf)

    def _train_vision_data(self, data):
        # Loading dataset
        training_set = validation_set = prediction_test_set = list()

        try:
            train_generator = ImageDataGenerator(rescale=1. / 255,
                                                 shear_range=0.5,
                                                 zoom_range=0.5,
                                                 rotation_range=7,
                                                 dtype='float16')
            training_set = train_generator.flow_from_directory(
                os.path.join(self.conf.working_path, data, ClassifyFactory.TRAINING),
                target_size=(self.conf.nn_image_size, self.conf.nn_image_size),
                color_mode='grayscale',
                batch_size=self.conf.nn_batch_size,
                class_mode='categorical')

            validation_generator = ImageDataGenerator(rescale=1. / 255, dtype='float16')
            validation_set = validation_generator.flow_from_directory(
                os.path.join(self.conf.working_path, data, ClassifyFactory.VALIDATION),
                color_mode='grayscale',
                target_size=(self.conf.nn_image_size, self.conf.nn_image_size),
                batch_size=self.conf.nn_batch_size,
                class_mode='categorical')

            prediction_test_set = self._predict_vision_data(data, suffix=ClassifyFactory.TEST)

        except Exception as e:
            self.logger.Error('VisionClass - Loading vision data', sys.exc_info())

        return training_set, validation_set, prediction_test_set

    def train_by(self, model, data):
        self._initialize()

        stats = Stats(classifier=model)
        training_set, validation_set, test_set = self._train_vision_data(data)

        if model == ClassifyFactory.NN_NETWORK:
            self.logger.Information(f'VisionClass - training {model}...')
            response, history = self.classify.launch_cnn_network(training_set, validation_set, test_set, True)
            stats.history = history
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be: NN_NETWORK'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    def train(self, model, data):
        self._initialize()

        stats = Stats(classifier=model)
        training_set, validation_set, prediction_set = self._train_vision_data(data)

        if model == ClassifyFactory.NN_NETWORK:
            self.logger.Information(f'TextClass - training {model}...')
            response, history = self.classify.launch_cnn_network(
                training_set, validation_set, prediction_set, True)
            stats.history = history
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be: NN_NETWORK'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    def _predict_vision_data(self, data, suffix=''):
        test_generator = ImageDataGenerator(rescale=1. / 255, dtype='float16')
        test_set = test_generator.flow_from_directory(
            os.path.join(self.conf.working_path, data, suffix),
            target_size=(self.conf.nn_image_size, self.conf.nn_image_size),
            color_mode='grayscale',
            batch_size=self.conf.nn_batch_size,
            class_mode=None,
            shuffle=False)

        return test_set

    def predict_by(self, model, data):
        self._initialize()

        stats = Stats(classifier=model)
        prediction_set = self._predict_vision_data(data)

        if model == ClassifyFactory.NN_NETWORK:
            self.logger.Information(f'TextClass - {model} prediction...')
            response, history = self.classify.launch_cnn_network(
                None, None, prediction_set, False)
            self.classify.show_metrics(None, response, stats=stats)
            stats.history = history
        else:
            response = []
            stats.info = 'Unsuitable prediction model. ' \
                         'Should be: NN_NETWORK'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    def predict(self, model, data):
        self._initialize()

        stats = Stats(classifier=model)
        prediction_set = self._predict_vision_data(data)

        if model == ClassifyFactory.NN_NETWORK:
            self.logger.Information(f'TextClass - {model} prediction...')
            response, history = self.classify.launch_cnn_network(
                None, None, prediction_set, False)
            self.classify.show_metrics(None, response, stats=stats)
            stats.history = history
        else:
            response = []
            stats.info = 'Unsuitable prediction model. ' \
                         'Should be: NN_NETWORK'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    @staticmethod
    def check_model(model):
        # is a available classification method
        if not (model == ClassifyFactory.NN_NETWORK):
            return False

        return True

    def check_source(self, data):
        if not ClassFile.has_media_file(os.path.join(self.conf.working_path, data)):
            return False

        return True
