import logging
import os
import sys
from queue import Queue
from threading import Thread

from common.model.ClassFile import ClassFile
from extract.DocToImg import DocToImg
from extract.DocToTxt import DocToTxt
from extract.Logger import Logger


class GbcProcessBatch:
    def __init__(self):
        self.logger = Logger(__name__, True)

    @staticmethod
    def from_file(file):
        file_filter = ClassFile.filter_by_size(file)

        file_path, file_name = os.path.split(file_filter)
        doc_to_image = DocToImg(file_path, file_path)

        img_list = doc_to_image.run(file_name)
        for image in img_list:
            doc_to_image.run(file_name, image)

        doc_to_txt = DocToTxt(file_path, file_path)
        doc_to_txt.run(file_name, save_to_file=True)

    @staticmethod
    def from_data(working_path, force=False):
        pdf_list = ClassFile.list_pdf_files(working_path)
        pdf_list_filter = ClassFile.filter_by_size(pdf_list)

        count = len(pdf_list_filter)
        for pdf_file_path in pdf_list_filter:
            file_path, file_name = os.path.split(pdf_file_path)
            doc_to_image = DocToImg(file_path, file_path)

            img_list = doc_to_image.run(file_name)
            for image in img_list:
                doc_to_image.run(file_name, image)

            doc_to_txt = DocToTxt(file_path, file_path)
            doc_to_txt.run(file_name, save_to_file=True)

            count -= 1
            print("LEFT: " + str(count))

    def _do_pre_process(self, process, q, result):
        """
        launch text processing in threads

        """
        while not q.empty():
            work = q.get()  # fetch new work from the Queue
            try:
                self.logger.Information("Requested..." + str(work[0]))
                data = process(work[1])
                result[work[0]] = data  # Store data back at correct index
                self.logger.Information("Done..." + str(work[0]))
            except Exception as e:
                result[work[0]] = None
                self.logger.Error('PreProcess::do_pre_process {}'.format(e), sys.exc_info())

            # signal to the queue that task has been processed
            q.task_done()
        return True

    def pre_process_batches(self, process, batch_list):
        """
        Process all txt files in batches to get the .grams

        """
        # categories = set()

        logging.info("GbcProcessBatch::Asynchronous tool")
        all_docs = [None for _ in batch_list]
        q = Queue(maxsize=0)

        counter = 0
        total = len(batch_list)
        i = 0

        while i < total:
            h = i
            for j in range(50):
                if h < total:
                    f = batch_list[h]
                    self.logger.Information('doc %s to q' % (counter + 1))
                    q.put((counter, f))

                    counter += 1
                h += 1

            for j in range(q.qsize()):
                worker = Thread(target=self._do_pre_process, args=(process, q, all_docs))
                worker.setDaemon(True)
                worker.start()

            # now we wait until the queue has been processed
            q.join()

            q.empty()
            i = h

        # self.logger.Information(len(categories), categories)
