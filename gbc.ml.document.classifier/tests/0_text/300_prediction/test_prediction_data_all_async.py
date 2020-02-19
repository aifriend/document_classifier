import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoPrediction(unittest.TestCase):

    def test_35_prediction_all_by(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                source=ClassifyService.FOLDER,
                action=ClassifyService.PREDICT,
                mode=ClassifyService.TEXT,
                data="text_data",
                domain=r"D:/Testing/test/predict/svh",
                model=ClassifyService.ALL_BY).status_code == 200
        )


def suite_prediction():
    suite = unittest.TestSuite()
    suite.addTest(TestDoPrediction("test_35_prediction_all_by"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_prediction())
