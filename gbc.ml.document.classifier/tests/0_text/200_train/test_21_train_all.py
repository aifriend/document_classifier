import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoTraining(unittest.TestCase):

    def test_211_training_decision_tree_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.DECISION_TREE).status_code == 200
        )

    def test_212_training_extra_tree_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.EXTRA_TREES).status_code == 200
        )

    def test_213_training_bayes_multi_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.NAIVE_BAYES_MULTI).status_code == 200
        )

    def test_214_training_bayes_complement_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.NAIVE_BAYES_COMPLEMENT).status_code == 200
        )

    def test_215_training_random_forest_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.RANDOM_FOREST).status_code == 200
        )

    def test_216_training_bagging_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.BAGGING).status_code == 200
        )

    def test_217_training_boosting_ada_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.BOOSTING_ADA).status_code == 200
        )

    def test_218_training_boosting_sgd_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.BOOSTING_SGD).status_code == 200
        )


def suite_train():
    suite = unittest.TestSuite()
    suite.addTest(TestDoTraining("test_211_training_decision_tree_by_text"))
    suite.addTest(TestDoTraining("test_212_training_extra_tree_by_text"))
    suite.addTest(TestDoTraining("test_213_training_bayes_multi_by_text"))
    suite.addTest(TestDoTraining("test_214_training_bayes_complement_by_text"))
    suite.addTest(TestDoTraining("test_215_training_random_forest_by_text"))
    suite.addTest(TestDoTraining("test_216_training_bagging_by_text"))
    suite.addTest(TestDoTraining("test_217_training_boosting_ada_by_text"))
    suite.addTest(TestDoTraining("test_218_training_boosting_sgd_by_text"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_train())
