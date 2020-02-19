import os
import re

from pdf2image import convert_from_path
from tika import parser

from extract.Logger import Logger


class Pdf:

    @staticmethod
    def to_image(pdf_list):
        for pdf in pdf_list:
            pages = convert_from_path(pdf, 500)
            file_root, file_extension = os.path.splitext(pdf)
            pages[0].save(file_root + '.jpg')

    @staticmethod
    def to_text(pdf_list):
        logger = Logger(__name__, True)
        count = 0
        for pdf in pdf_list:
            # Parse data from file
            file_data = parser.from_file(pdf)
            # Get files text content
            text = file_data['content']

            logger.Information(pdf)
            text = Pdf.clean_text(text)
            logger.Information(text)

            file_root, file_extension = os.path.splitext(pdf)
            if text is not None and len(text) > 50:
                try:
                    new_file = open(file_root + '.txt', mode="w", encoding="utf-8")
                    new_file.write(text)
                    new_file.flush()
                    new_file.close()
                except Exception as e:
                    logger.Error(e)
                    pass

            count += 1

        logger.Information(str(count) + " processed!")

    @staticmethod
    def clean_text(text):
        if text is None:
            return ''

        new_ = re.sub(r'\n\n', r'\n', text)
        new_ = re.sub(r'\t', r' ', new_)
        new_ = re.sub(r' {2}', r' ', new_)
        new_ = re.sub(r'--', r'-', new_)

        return new_.strip()
