import sys

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from classifier.IClassify import IClassify


class NaiveBayes(IClassify):
    MULTINOMIAL = "MULTINOMIAL"
    COMPLEMENT = "COMPLEMENT"

    def __init__(self, conf):
        super().__init__(conf)

    def initialize(self, subtype=MULTINOMIAL):
        if subtype is NaiveBayes.COMPLEMENT:
            self.clf = ComplementNB()
        elif subtype is NaiveBayes.MULTINOMIAL:
            self.clf = MultinomialNB()
        else:
            self.logger.Error("Unknown Naive-Bayes type", sys.exc_info())

    def train(self, x, y, test, subtype=MULTINOMIAL, s3Service=None):
        self.initialize(subtype=subtype)
        self.do_train(x, y)
        if s3Service:
            self.save_model_s3(self.conf.nb_model.replace('(subtype)', subtype), s3Service)
        else:
            self.save_model(self.conf.nb_model.replace('(subtype)', subtype))
        response = self.get_prediction(test)
        return response

    def predict(self, x, subtype=MULTINOMIAL, s3Service=None):
        self.load_model(subtype=subtype, s3Service=s3Service)
        response = self.get_prediction(x)
        return response

    def predict_prob(self, x, subtype=MULTINOMIAL):
        self.load_model(subtype=subtype)
        classes, prob = self.do_predict_prob(x)
        return classes, prob

    def load_model(self, subtype=MULTINOMIAL, s3Service=None):
        super().load_model(self.conf.nb_model.replace('(subtype)', subtype), s3Service)
