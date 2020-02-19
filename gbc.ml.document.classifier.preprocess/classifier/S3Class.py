import json
import sys
from queue import Queue
from threading import Thread

import httpx

from Configuration import Configuration
from commonsLib import loggerElk


class S3Class:

    def __init__(self, conf: Configuration):
        self.conf = conf
        self.logger = loggerElk(__name__, True)

    def get_text_from_s3(self, domain, bucket, s3_session):
        resp = s3_session.list_objects_v2(Bucket=bucket, Prefix=domain)
        url_pdf_to_readable = self.conf.pdf_to_readable_url
        q = Queue(maxsize=0)
        all_docs = dict()
        for obj in resp['Contents']:
            if obj['Size'] > 0 and obj['Key'].upper().endswith('.PDF') and not obj['Key'].upper().endswith(
                    'READABLE.PDF'):
                has_txt_list = [x for x in resp['Contents']
                                if
                                x['Key'].upper().startswith(obj['Key'].upper()) and x['Key'].upper().endswith(".TXT")]
                if len(has_txt_list) == 0:
                    key = obj['Key']

                    all_docs[key] = None

        i = 0
        counter = 0
        keys_array = list(all_docs.keys())
        while i < len(keys_array):
            h = i
            for j in range(self.conf.max_threads_pdf_2_readable):
                if h < len(keys_array):
                    q.put((keys_array[counter], bucket, url_pdf_to_readable))

                    counter += 1
                h += 1
            for j in range(q.qsize()):
                worker = Thread(target=self._do_pdf_2_readable_preprocess, args=(q, all_docs))
                worker.setDaemon(True)  # setting threads as "daemon" allows main program to
                # exit eventually even if these dont finish
                # correctly.
                worker.start()
            # now we wait until the queue has been processed
            q.join()

            q.empty()
            i = h
        self.logger.Information('GbcMlDocumentClassifier::TextFactory - get_text_from_s3...')
        return json.dumps(all_docs)

    def _do_pdf_2_readable_preprocess(self, q, result):  # q:[[index, text, kind, path], ...]
        """
        launch text processing in threads
        """
        while not q.empty():
            work = q.get()  # fetch new work from the Queue
            try:
                self.logger.Information("Requested..." + str(work[0]))
                data = self._process_pdf_to_readable(work[0], work[1], work[2])
                result[work[0]] = data  # Store data back at correct index
                self.logger.Information("Done..." + str(work[0]))
            except Exception as e:
                self.logger.Error('PreProcess::_do_pdf_2_readable_preprocess::{}'.format(e), sys.exc_info())

            # signal to the queue that task has been processed
            q.task_done()
        return True

    def _process_pdf_to_readable(self, key, bucket, url_pdf_to_readable):
        data = {
            'source': 'S3',
            'data': key,
            'key': key,
            'persistence': 'S3',
            'lang': 'spa',
            'bucket': bucket
        }
        header = {"Content-Type": "application/json"}
        with httpx.Client() as client:
            resp_pdf_to_readable = client.post(
                url_pdf_to_readable, data=json.dumps(data), headers=header, timeout=2000)

        if resp_pdf_to_readable.status_code == 200:
            return json.loads(resp_pdf_to_readable.text)
        else:
            self.logger.Error(
                f"Error with document {key} -> {resp_pdf_to_readable.status_code} {resp_pdf_to_readable.text}")
            return {"key": key, "error": resp_pdf_to_readable.text, "code": resp_pdf_to_readable.status_code}
