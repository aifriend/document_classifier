from sklearn.ensemble import VotingClassifier

from classifier.IClassify import IClassify


class Voting(IClassify):

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self, clf_list):
        self.clf = VotingClassifier(
            estimators=clf_list,
            voting=self.conf.voting,
            n_jobs=self.conf.voting_n_jobs)

    def train(self, x, y, test, class_list):
        self.initialize(class_list)
        self.do_train(x, y)
        self.save_model(class_list)
        response = self.get_prediction(test)
        return response

    def predict(self, x, class_list):
        self.initialize(class_list)
        self.load_model(class_list)
        response = self.get_prediction(x)
        return response

    def predict_prob(self, x, class_list):
        self.initialize(class_list)
        self.load_model(class_list)
        classes, prob = self.do_predict_prob(x)
        return classes, prob

    def save_model(self, class_list=''):
        # file_name = "_".join([str(n[0]) for n in class_list])
        model_name = self.conf.voting_model  # + "_" + file_name
        super().save_model(model_name)

    def load_model(self, class_list=''):
        # file_name = "_".join([str(n[0]) for n in class_list])
        model_name = self.conf.voting_model  # + "_" + file_name
        super().load_model(model_name)
