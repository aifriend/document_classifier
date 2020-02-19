import os
import pickle
import sys
import time

import pandas as pd
import win32com.client
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from tika import parser

from extract.DocToImg import DocToImg
from extract.ImgToTxt import ImgToTxt
from extract.Logger import Logger


class DocToTxt:
    def __init__(self, path_input, path_output):
        self.doc_to_image = DocToImg(path_input, path_output)
        self.path_in = path_input
        self.path_out = path_output
        self.logger = Logger(__name__, True)

    def _to_file_df(self, file_name, full_df):
        file_root, file_extension = os.path.splitext(file_name)
        path_to_doc = os.path.join(self.path_out, file_root, (file_name + '.pkl'))
        if not os.path.isfile(path_to_doc):
            with open(path_to_doc, 'wb') as f:
                pickle.dump(full_df, f)

    @staticmethod
    def _pdf_miner(file, max_pages):
        pdf_parser = PDFParser(file)
        doc = PDFDocument(pdf_parser)
        pdf_parser.set_document(doc)
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        laparams.char_margin = 1.0
        laparams.word_margin = 1.0
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        todo_texto = ''

        num_page = 0
        for i, page in enumerate(PDFPage.create_pages(doc)):
            num_page += 1
            if num_page >= max_pages:
                break
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    todo_texto += lt_obj.get_text()
                # elif isinstance(lt_obj, LTImage):
                #     print("imagen encontrada")
                #     imgbits = lt_obj.bits
                #     img_data = lt_obj.stream.get_data()
                # elif isinstance(lt_obj, LTFigure):
                #     print("figura encontrada")
                #     fig_data = lt_obj.stream.get_data()
                # elif isinstance(lt_obj, LTCurve):
                #     print("curva encontrada")
                #     curva = lt_obj.get_pts()

        if len(todo_texto) >= 50:
            return todo_texto
        else:
            raise

    @staticmethod
    def _tiker(file):
        all_text = parser.from_file(file)['content']
        keep_looking = False
        keep_looking |= ' ' not in all_text
        keep_looking |= len(all_text.strip()) <= 100

        if not keep_looking:
            return all_text
        else:
            raise

    def _pdf_to_txt(self, file_name, max_pages, native=False, page_to_file=True, to_panda_file=False):
        if native:
            f = open(os.path.join(self.path_in, file_name), 'rb')
            try:
                # return self._pdf_miner(f, max_pages)
                return DocToTxt._tiker(f)
            except Exception as e:
                self.logger.Error('DocToTxt::run::{}'.format(e), sys.exc_info())

            finally:
                f.close()

        img_list = self.doc_to_image.run(file_name)
        todo_texto = ''
        img_to_txt = ImgToTxt()
        num_page = 0
        for idx, img in enumerate(img_list):
            string_txt = img_to_txt.get_string(file_name, idx, img)
            if page_to_file:
                self._page_to_file(file_name, idx, string_txt)
            todo_texto += string_txt
            num_page += 1
            if num_page >= max_pages:
                break

        if to_panda_file:
            text_df_list = []
            img_to_txt = ImgToTxt()
            num_page = 0
            for idx, img in enumerate(img_list):
                doc_txt = img_to_txt.get_doc(file_name, idx, img)
                text_df_list.append(doc_txt)
                num_page += 1
                if num_page >= max_pages:
                    break
            full_df = pd.concat(text_df_list, 0)
            self._to_file_df(file_name, full_df)

        return todo_texto

    def _doc_to_txt(self, file_name):
        app = win32com.client.Dispatch('Word.Application')
        # app.Visible = True
        doc = app.Documents.Open(os.path.join(self.path_in, file_name))
        content = doc.Content.Text
        app.Quit()

        return content

    def _excel_to_txt(self, file_name):
        pd.set_option('max_colwidth', 200)
        xl = pd.ExcelFile(os.path.join(self.path_in, file_name))
        sheet_names = xl.sheet_names
        todo_texto = ''
        for sheet in sheet_names:
            datos = xl.parse(sheet)
            if len(datos.get_values) == 0:
                continue
            todo_texto += pd.DataFrame.to_string(datos, header=False, index=False, na_rep='') + '\n'

        return todo_texto

    def _document_to_text(self, file_name, max_pages, forced="PDF"):
        file = os.path.join(self.path_in, file_name)
        ind_punto = str.rfind(file, '.')
        extension = file[ind_punto:]
        extension = extension.lower()

        try:
            if forced == 'PDF' and extension == '.pdf':
                return self._pdf_to_txt(file_name, max_pages, page_to_file=False)
            elif forced == 'DOC' and ((extension == '.doc') or (extension == '.docx')):
                return self._doc_to_txt(file_name)
            elif forced == 'XLS' and ((extension == '.xls') or (extension == '.xlsx')):
                return self._excel_to_txt(file_name)
            else:
                self.logger.Debug("DocToTxt::right source file format not found!")
        except Exception as e:
            self.logger.Error('DocToTxt::document_to_text::{}'.format(e), sys.exc_info())
            return ''

    def _save_text_file(self, file_name, content, subdir=False):
        if isinstance(content, str):
            file_root, file_extension = os.path.splitext(file_name)
            path_to_doc = os.path.join(self.path_out, (file_root if subdir else ''))
            if not os.path.isdir(path_to_doc):
                os.mkdir(path_to_doc)
            file = os.path.join(path_to_doc, (file_name + r".txt"))
            if not os.path.isfile(file):
                try:
                    text_file = open(file, "w")
                    text_file.write(content)
                    text_file.close()
                except Exception as e:
                    self.logger.Error('DocToTxt::save_text_file::{}'.format(e), sys.exc_info())

    def _page_to_file(self, file_name, idx, content):
        idx += 1
        file_root, file_extension = os.path.splitext(file_name)
        path_to_doc = os.path.join(
            self.path_out, file_root, (file_name + "_" + str(idx).zfill(2) + r'.txt'))
        if not os.path.isfile(path_to_doc):
            try:
                with open(path_to_doc, 'w') as f:
                    f.write(content)
                    f.close()
            except Exception as e:
                self.logger.Error('DocToTxt::page_to_file::{}'.format(e), sys.exc_info())

    def has_text_file(self, file_name):
        file_root, file_extension = os.path.splitext(file_name)
        path_to_doc = os.path.join(self.path_out, file_root)
        if os.path.isdir(path_to_doc):
            file = os.path.join(path_to_doc, (file_name + '.pkl'))
            if os.path.isfile(file):
                return True

        return False

    def get_text(self, path_input, max_pages=sys.maxsize):
        texto = self._document_to_text(path_input, max_pages)
        return texto

    def run(self, pdf_file_name, max_pages=sys.maxsize, save_to_file=False):
        start = time.time()
        texto = self._document_to_text(pdf_file_name, max_pages)
        stop = time.time()
        print("La conversion general del DOC [{0}] ha tardado "
              .format(pdf_file_name) + str(round(stop - start, 3)) + " segundos\n")

        if save_to_file:
            self._save_text_file(pdf_file_name, texto)

        return texto
