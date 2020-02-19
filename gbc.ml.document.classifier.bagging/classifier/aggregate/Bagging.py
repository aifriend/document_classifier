from sklearn.ensemble import BaggingClassifier

from classifier.IClassify import IClassify


class Bagging(IClassify):

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self):
        self.clf = BaggingClassifier(
            n_estimators=self.conf.bagging_n_estimators,
            max_samples=self.conf.bagging_max_samples,
            n_jobs=self.conf.bagging_n_jobs,
            verbose=self.conf.bagging_verbose)

    def train(self, x, y, test):
        self.initialize()
        self.do_train(x, y)
        self.save_model(self.conf.bagging_model)
        response = self.get_prediction(test)
        return response

    def predict(self, x):
        self.load_model()
        response = self.get_prediction(x)
        return response

    def predict_prob(self, x):
        self.load_model()
        classes, prob = self.do_predict_prob(x)
        return classes, prob

    def load_model(self, path=''):
        super().load_model(self.conf.bagging_model)
