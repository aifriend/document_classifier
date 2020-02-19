from sklearn.tree import DecisionTreeClassifier

from classifier.IClassify import IClassify


class DecisionTree(IClassify):

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self):
        self.clf = DecisionTreeClassifier(
            max_depth=self.conf.dt_max_depth
        )

    def train(self, x, y, test, s3_service=None):
        self.initialize()
        self.do_train(x, y)
        if s3_service:
            self.save_model_s3(self.conf.dt_model, s3_service)
        else:
            self.save_model(self.conf.dt_model)
        response = self.get_prediction(test)
        return response

    def predict(self, x, s3_service=None):
        self.load_model(s3_service=s3_service)
        response = self.get_prediction(x)
        return response

    def predict_prob(self, x):
        self.load_model()
        classes, prob = self.do_predict_prob(x)
        return classes, prob

    def load_model(self, path='', s3_service=None):
        super().load_model(self.conf.dt_model, s3_service)
