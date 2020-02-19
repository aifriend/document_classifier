import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoTraining(unittest.TestCase):

    def test_training_bayes_multi_by_text(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                action=ClassifyService.TRAIN,
                source=ClassifyService.S3,
                mode=ClassifyService.TEXT,
                domain=r"servihabitat/AF-07-NOTS-01",
                bucket="gbc.ml.document.classifier",
                model=ClassifyService.NAIVE_BAYES_MULTI).status_code == 200
        )

    # def test_training_bayes_complement_by_text(self):
    #     self.assertTrue(
    #         ITest.do_request(
    #             server_url=os.path.join(CLIENT_URL, conf.client_training),
    #             action=ClassifyService.TRAIN,
    #             source=ClassifyService.S3,
    #             mode=ClassifyService.TEXT,
    #             domain=r"AF-03-CERA-19",
    #             bucket="gbc.ml.document.classifier",
    #             model=ClassifyService.NAIVE_BAYES_COMPLEMENT).status_code == 200
    #     )


def suite_train():
    suite = unittest.TestSuite()
    suite.addTest(TestDoTraining("test_training_bayes_multi_by_text"))
    # suite.addTest(TestDoTraining("test_training_bayes_complement_by_text"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_train())
