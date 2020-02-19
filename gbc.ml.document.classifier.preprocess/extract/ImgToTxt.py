import logging
import os
import sys
import time

import pandas as pd
import pytesseract as tess

from extract.Logger import Logger


class ImgToTxt:

    def __init__(self):
        self.logger = self.logger = Logger(__name__, True)

    @staticmethod
    def __img_to_string(img):
        lang = 'spa'
        config = ('-l ' + lang + ' --oem 1 --psm 3')
        tess.pytesseract.tesseract_cmd = r"tesseract"

        start = time.time()
        text_string = tess.image_to_string(img, config=config)
        stop = time.time()
        print("Tesseract ha tardado " + str(round(stop - start, 3)) + " en convertir esta pagina")

        return text_string

    @staticmethod
    def __img_to_doc(img):
        lang = 'spa'
        config = ('-l ' + lang + ' --oem 1 --psm 3')
        tess.pytesseract.tesseract_cmd = r"tesseract"
        # start = time.time()
        # text_string = tess.image_to_string(img, config=config, output_type="string")
        #
        # stop = time.time()
        # print("Tesseract ha tardado " + str(round(stop - start, 3)) + " en convertir esta pagina")

        start = time.time()
        text_dict = tess.image_to_data(img, config=config, output_type="dict")
        text_df = pd.DataFrame.from_dict(text_dict)
        # Quedarnos solo con las fiables
        text_df_conf = text_df.loc[(text_df['conf']).astype(int) > 10]
        stop = time.time()
        print("Tesseract ha tardado " + str(round(stop - start, 3)) + " en convertir esta pagina")

        # # text_df.to_csv('archivo_csv.xls', sep='\t')
        # text_string2 = ''
        # for token in text_dict['text']:
        #     text_string2 += token + ' '

        return text_df_conf

    def __save_text_file(self, file_path, content):
        try:
            if not os.path.isfile(file_path):
                text_file = open(file_path, "w")
                text_file.write(content)
                text_file.close()
        except Exception as e:
            self.logger.Error('ImgToTxt::save_text_file::{}'.format(e), sys.exc_info())

    def get_string(self, file_name, idx, file_input):
        start = time.time()
        texto = self.__img_to_string(file_input)
        stop = time.time()
        print("La conversion to TXT-STRING [{0} -> IMG {1}] ha tardado "
              .format(file_name, idx) + str(round(stop - start, 3)) + " segundos\n")

        return texto

    def get_doc(self, file_name, idx, file_input):
        start = time.time()
        texto = self.__img_to_doc(file_input)
        stop = time.time()
        print("La conversion to TXT-DOC [{0} -> IMG {1}] ha tardado "
              .format(file_name, idx) + str(round(stop - start, 3)) + " segundos\n")

        return texto

    def run(self, path_input, path_output):
        logging.info("ImageToTxt::Processing data - Tesseract")
        texto = self.__img_to_doc(path_input)
        self.__save_text_file(path_output, texto)
