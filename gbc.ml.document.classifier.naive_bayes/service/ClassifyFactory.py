import sys
import warnings

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from classifier.basic.NaiveBayes import NaiveBayes
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

    # classification model type
    NAIVE_BAYES_MULTI = "NAIVE_BAYES_MULTI"
    NAIVE_BAYES_COMPLEMENT = "NAIVE_BAYES_COMPLEMENT"

    def __init__(self, conf):
        self.conf = conf
        self._naive_bayes = NaiveBayes(conf)
        self.logger = loggerElk(__name__, True)

    @staticmethod
    def encode_categories(y):
        # encode class values as integers
        encoder = LabelEncoder()
        encoded_Y = encoder.fit_transform(y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = to_categorical(encoded_Y).astype(int)
        return dummy_y

    def show_metrics(self, model, y_test, y_predicted, stats=None):
        if y_test is None:
            self.logger.Information(f'Predicted {str.lower(model)}: ' + str(y_predicted))
            # stats.info = y_predicted
        elif y_predicted is not None and len(y_predicted) > 0:
            self.logger.Information(f'Accuracy {str.lower(model)}: '
                                    + str(accuracy_score(y_test, y_predicted[0])))
            # self.logger.Information(metrics.classification_report(y_test, y_predicted[0]))
            stats.info = metrics.classification_report(y_test, y_predicted[0])
        return stats

    def launch_naive_bayes_complement(self, X_train=None, y_train=None, X_test=None, train=False, s3Service=None):
        self.logger.Information('---< Naive-Bayes Complement >---')
        result = ''
        if train and X_train is not None and y_train is not None and X_test is not None:
            result = self._naive_bayes.train(X_train, y_train, X_test, NaiveBayes.COMPLEMENT, s3Service)
        elif not train and X_test is not None:
            result = self._naive_bayes.predict(X_test, NaiveBayes.COMPLEMENT)

        return result

    def launch_naive_bayes_multinomial(self, X_train=None, y_train=None, X_test=None, train=False, s3Service=None):
        self.logger.Information('---< Naive-Bayes Multinomial >---')
        result = ''
        if train and X_train is not None and y_train is not None and X_test is not None:
            result = self._naive_bayes.train(X_train, y_train, X_test, NaiveBayes.MULTINOMIAL, s3Service)
        elif not train and X_test is not None:
            result = self._naive_bayes.predict(X_test, NaiveBayes.MULTINOMIAL, s3Service)

        return result
