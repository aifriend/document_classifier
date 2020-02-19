import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import CLIENT_URL, conf

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoTraining(unittest.TestCase):

    def test_training_extra_tree_by_text(self):
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


def suite_train():
    suite = unittest.TestSuite()
    suite.addTest(TestDoTraining("test_training_extra_tree_by_text"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_train())
