import logging
import os
import unittest

from classifier.ClassifyService import ClassifyService
from tests.ITest import ITest
from tests.helper import conf, CLIENT_URL

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class TestDoPreProcess(unittest.TestCase):

    def test_pre_process_folder_gram(self):
        self.assertTrue(
            ITest.do_request(
                server_url=os.path.join(CLIENT_URL, conf.client_pre_process),
                action=ClassifyService.PRE_PROCESS,
                source=ClassifyService.S3,
                mode=ClassifyService.GRAM,
                force=True,
                domain=r"AF-03-CERA-19",
                bucket="gbc.ml.document.classifier").status_code == 200
        )


def suite_fit():
    suite = unittest.TestSuite()
    suite.addTest(TestDoPreProcess('test_pre_process_folder_gram'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_fit())
