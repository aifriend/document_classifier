import sys
import warnings

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from classifier.basic.NNetwork import NNetwork
from common.model.ClassFile import ClassFile
from commonsLib import loggerElk

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def get_category(gram_path):
    return ClassFile.get_containing_dir_name(gram_path)


class ClassifyFactory:
    TIMEOUT = sys.maxsize

    # source type
    PLAINTEXT = "PLAINTEXT"
    FOLDER = "FOLDER"
    S3 = "S3"

    # mode type
    TEXT = "TEXT"
    IMAGE = "IMAGE"

    # action type
    PRE_PROCESS = "PRE_PROCESS"
    TRAIN = "TRAIN"
    PREDICT = "PREDICT"

    # execution mode
    NORMAL = "NORMAL"

    # classification model type
    NN_NETWORK = "NN_NETWORK"

    # machine learning training sources
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "testing"

    def __init__(self, conf):
        self.conf = conf
        self._cnn_network = NNetwork(conf)
        self.logger = loggerElk(__name__, True)

    @staticmethod
    def encode_categories(y):
        # encode class values as integers
        encoder = LabelEncoder()
        encoded_Y = encoder.fit_transform(y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = to_categorical(encoded_Y).astype(int)
        return dummy_y

    def show_metrics(self, y_test, y_predicted, stats=None):
        if y_test is None:
            self.logger.Information('Predicted: ' + str(y_predicted))
            stats.info = y_predicted
        elif y_predicted is not None and len(y_predicted) > 0:
            self.logger.Information('Accuracy: ' + str(accuracy_score(y_test, y_predicted[0])))
            # self.logger.Information(metrics.classification_report(y_test, y_predicted[0]))
            stats.info = metrics.classification_report(y_test, y_predicted[0])
        return stats

    def launch_cnn_network(self, training_set=None, validation_set=None, prediction_set=None, train=False):
        self.logger.Information('---< Nn Network >---')
        result = ''
        history = ''
        if train and training_set is not None and validation_set is not None and prediction_set is not None:
            result, history = self._cnn_network.train(training_set, validation_set, prediction_set)
        elif not train and prediction_set is not None:
            result = self._cnn_network.predict(prediction_set)

        return result, history
