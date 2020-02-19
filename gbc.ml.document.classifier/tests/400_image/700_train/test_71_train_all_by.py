import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoTraining(unittest.TestCase):

    def test_image_training_all(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_training),
                source=ClassifyService.FOLDER,
                action=ClassifyService.TRAIN,
                mode=ClassifyService.IMAGE,
                data="image_data",
                domain=r"D:/Testing/test/train/svh",
                model=ClassifyService.ALL_BY) is not None
        )


def suite_train():
    suite = unittest.TestSuite()
    suite.addTest(TestDoTraining("test_image_training_all"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_train())
