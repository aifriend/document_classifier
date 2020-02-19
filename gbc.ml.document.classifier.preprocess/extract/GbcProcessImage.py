import logging
import os
from random import shuffle

from common.model.ClassFile import ClassFile
from extract.DocToImg import DocToImg
from extract.GbcProcessBatch import GbcProcessBatch


class GbcProcessImage(GbcProcessBatch):

    def __init__(self, conf):
        self.conf = conf
        super().__init__()

    @staticmethod
    def from_data(working_path, force=False):
        logging.info("GbcProcessImage::Processing data - From PDF to JPG")
        pdf_list = ClassFile.list_pdf_files(working_path)
        pdf_list_filter = ClassFile.filter_by_size(pdf_list)
        if not force:
            pdf_list_filter = ClassFile.filter_duplicate(
                working_path, pdf_list_filter, "jpg")

        count = len(pdf_list_filter)
        for pdf_file_path in pdf_list_filter:
            file_path, file_name = os.path.split(pdf_file_path)
            doc_to_image = DocToImg(file_path, file_path)
            doc_to_image.run(file_name, save_to_file=True)
            count -= 1
            print("LEFT: " + str(count))

    @staticmethod
    def _process(file_dir):
        file_path, file_name = os.path.split(file_dir)
        doc_to_image = DocToImg(file_path, file_path)
        doc_to_image.run(file_name, save_to_file=True)

    def create_data_async(self, working_path=None, force=False):
        logging.info("GbcProcessImage::Processing data - From PDF to JPG - asynchronous")
        working_path = working_path if working_path else self.conf.working_path
        file_list = ClassFile.list_pdf_files(working_path)
        file_size_filter = ClassFile.filter_by_size(file_list)
        if not force:
            file_size_filter = ClassFile.filter_duplicate(
                working_path, file_size_filter, self.conf.image_file_ext)
        shuffle(file_size_filter)
        print(f"Processing: {len(file_size_filter)} documents")
        self.pre_process_batches(self._process, file_size_filter)

    def process_file(self, file):
        logging.info("GbcProcessText::Processing file - From PDF to JPG")
        doc_to_txt = DocToImg(self.conf.working_path, self.conf.working_path)
        file_root = ClassFile.get_file_root_name(file)
        file = file_root + "." + self.conf.source_file_ext
        doc_to_txt.run(file, save_to_file=True)
