import logging
import os

from common.model.ClassFile import ClassFile
from extract.DocToImg import DocToImg
from extract.DocToTxt import DocToTxt
from extract.GbcProcessBatch import GbcProcessBatch


class GbcProcessAll(GbcProcessBatch):

    def __init__(self):
        super().__init__()

    @staticmethod
    def create_data(working_path):
        logging.info("GbcProcessAll::Processing data - From PDF to TXT and JPG")
        file_list = ClassFile.list_pdf_files(working_path)
        file_filter = ClassFile.filter_by_size(file_list)

        count = len(file_filter)
        for source_path in file_filter:
            GbcProcessAll._process(source_path)
            count -= 1
            print("LEFT: " + str(count))

    @staticmethod
    def _process(file_dir):
        file_path, file_name = os.path.split(file_dir)
        doc_to_image = DocToImg(file_path, file_path)

        img_list = doc_to_image.run(file_name)
        for image in img_list:
            doc_to_image.run(file_name, image)

        doc_to_txt = DocToTxt(file_path, file_path)
        doc_to_txt.run(file_name, save_to_file=True)

    def create_data_async(self, working_path, force=False):
        logging.info("GbcProcessAll::Processing data - From PDF to TXT and JPG")
        file_list = ClassFile.list_pdf_files(working_path)
        file_filter = ClassFile.filter_by_size(file_list)
        if not force:
            file_filter = ClassFile.filter_duplicate(
                working_path, file_filter, "jpg")
            file_filter = ClassFile.filter_duplicate(
                working_path, file_filter, "txt")
        self.pre_process_batches(self._process, file_filter)
