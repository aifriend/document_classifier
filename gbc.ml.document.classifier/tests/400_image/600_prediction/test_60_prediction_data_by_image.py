import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoPrediction(unittest.TestCase):

    def test_image_data_prediction(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_predict),
                action=ClassifyService.PREDICT,
                source=ClassifyService.FOLDER,
                mode=ClassifyService.IMAGE,
                data="image_data",
                domain=r"D:/Testing/data/predict/svh",
                model=ClassifyService.NN_NETWORK).status_code == 200
        )


def suite_prediction():
    suite = unittest.TestSuite()
    suite.addTest(TestDoPrediction("test_image_data_prediction"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_prediction())
