from sklearn.ensemble import ExtraTreesClassifier

from classifier.IClassify import IClassify


class ExtraTree(IClassify):

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self):
        self.clf = ExtraTreesClassifier(
            n_estimators=self.conf.et_n_estimators,
            max_features=self.conf.et_max_features,
            bootstrap=self.conf.et_bootstrap,
            n_jobs=self.conf.et_n_jobs,
            verbose=self.conf.et_verbose
        )

    def train(self, x, y, test):
        self.initialize()
        self.do_train(x, y)
        self.save_model(self.conf.et_model)
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
        super().load_model(self.conf.et_model)
