import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoPrediction(unittest.TestCase):

    def test_301_text_prediction_file_bagging(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.BAGGING).status_code == 200
        )

    def test_302_text_prediction_file_boosting_ada(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.BOOSTING_ADA).status_code == 200
        )

    def test_303_text_prediction_file_boosting_sgd(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.BOOSTING_SGD).status_code == 200
        )

    def test_304_text_prediction_file_decision_tree(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.DECISION_TREE).status_code == 200
        )

    def test_305_text_prediction_file_extra_trees(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.EXTRA_TREES).status_code == 200
        )

    def test_306_text_prediction_file_bayes_multi(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.NAIVE_BAYES_MULTI).status_code == 200
        )

    def test_307_text_prediction_file_bayes_complement(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.NAIVE_BAYES_COMPLEMENT).status_code == 200
        )

    def test_308_text_prediction_file_random_forest(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                file="PH14.txt",
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.RANDOM_FOREST).status_code == 200
        )


def suite_prediction():
    suite = unittest.TestSuite()
    suite.addTest(TestDoPrediction("test_301_text_prediction_file_bagging"))
    suite.addTest(TestDoPrediction("test_302_text_prediction_file_boosting_ada"))
    suite.addTest(TestDoPrediction("test_303_text_prediction_file_boosting_sgd"))
    suite.addTest(TestDoPrediction("test_304_text_prediction_file_decision_tree"))
    suite.addTest(TestDoPrediction("test_305_text_prediction_file_extra_trees"))
    suite.addTest(TestDoPrediction("test_306_text_prediction_file_bayes_multi"))
    suite.addTest(TestDoPrediction("test_307_text_prediction_file_bayes_complement"))
    suite.addTest(TestDoPrediction("test_308_text_prediction_file_random_forest"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_prediction())
