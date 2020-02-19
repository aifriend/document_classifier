import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoPrediction(unittest.TestCase):

    def test_312_text_prediction_file_boosting_ada(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.BOOSTING_ADA).status_code == 200
        )

    def test_313_text_prediction_file_boosting_sgd(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.BOOSTING_SGD).status_code == 200
        )


def suite_prediction():
    suite = unittest.TestSuite()
    suite.addTest(TestDoPrediction("test_312_text_prediction_file_boosting_ada"))
    suite.addTest(TestDoPrediction("test_313_text_prediction_file_boosting_sgd"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_prediction())
