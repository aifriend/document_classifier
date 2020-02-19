from sklearn.ensemble import RandomForestClassifier

from classifier.IClassify import IClassify


class RandomForest(IClassify):

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self):
        self.clf = RandomForestClassifier(
            n_estimators=self.conf.rf_n_estimators,
            max_leaf_nodes=self.conf.rf_max_leaf_nodes,
            n_jobs=self.conf.rf_n_jobs,
            verbose=self.conf.rf_verbose
        )

    def train(self, x, y, test):
        self.initialize()
        self.do_train(x, y)
        self.save_model(self.conf.rf_model)
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
        super().load_model(self.conf.rf_model)
