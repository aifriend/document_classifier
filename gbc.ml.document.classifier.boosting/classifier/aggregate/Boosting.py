import sys

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from classifier.IClassify import IClassify


class Boosting(IClassify):
    ADA = "ADA"
    SGD = "SGD"

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self, subtype):
        if subtype is Boosting.SGD:
            self.clf = GradientBoostingClassifier(
                n_estimators=self.conf.boosting_n_estimators,
                verbose=self.conf.boosting_verbose)
        elif subtype is Boosting.ADA:
            self.clf = AdaBoostClassifier(
                n_estimators=self.conf.boosting_n_estimators)
        else:
            self.logger.Error("Unknown Boosting type", sys.exc_info())

    def train(self, x, y, test, subtype):
        self.initialize(subtype=subtype)
        self.do_train(x, y)
        self.save_model(self.conf.boosting_model.replace('(subtype)', subtype))
        response = self.get_prediction(test)
        return response

    def predict(self, x, subtype):
        self.load_model(subtype=subtype)
        response = self.get_prediction(x)
        return response

    def predict_prob(self, x, subtype):
        self.load_model(subtype=subtype)
        classes, prob = self.do_predict_prob(x)
        return classes, prob

    def load_model(self, subtype):
        super().load_model(self.conf.boosting_model.replace('(subtype)', subtype))
