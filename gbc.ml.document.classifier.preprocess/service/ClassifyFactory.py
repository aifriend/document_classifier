import sys
import warnings

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from common.model.ClassFile import ClassFile
from commonsLib import loggerElk

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def get_category(gram_path):
    return ClassFile.get_containing_dir_name(gram_path)


class ClassifyFactory:
    TIMEOUT = sys.maxsize

    # source type
    FILE = "FILE"
    FOLDER = "FOLDER"
    VECTOR = "VECTOR"
    S3 = "S3"

    # mode type
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    GRAM = "GRAM"

    # action type
    PRE_PROCESS = "PRE_PROCESS"
    TRAIN = "TRAIN"
    PREDICT = "PREDICT"

    def __init__(self, conf):
        self.conf = conf
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
