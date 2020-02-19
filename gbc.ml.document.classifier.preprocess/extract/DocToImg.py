import os
import sys
import time

from pdf2image import convert_from_path

from extract.Logger import Logger


class DocToImg:
    def __init__(self, path_input, path_output):
        self.path_in = path_input
        self.path_out = path_output
        self.logger = Logger(__name__, True)

    def __pfd_to_img(self, file_name, max_pages=sys.maxsize):
        dpi = 200
        start = time.time()
        path = os.path.join(self.path_in, file_name)
        img = []
        try:
            images_from_path = convert_from_path(path, dpi=dpi, first_page=0, last_page=max_pages - 1)
            stop = time.time()
            print("\nEl PDF ha tardado en cargarse " + str(round(stop - start, 3)) + " segundos")
            i = 0
            for page in images_from_path:
                i += 1
                img.append(page.convert('RGB'))
                if i >= max_pages:
                    return img
        except Exception as _:
            print(f"\nEl documento: {path} no ha podido procesarse")
        return img

    @staticmethod
    def __save_img_file(file_path, file_name, image_content, idx, subdir=False):
        image_name = (file_name + '_' + str(idx).zfill(2) + '.jpg')
        path_to_image = os.path.join(file_path, (file_name if subdir else ''), image_name)
        if not os.path.isfile(path_to_image):
            image_content.save(path_to_image, quality=95)

    def run(self, pdf_file_name, max_pages=sys.maxsize, save_to_file=False):
        start = time.time()
        content = self.__pfd_to_img(pdf_file_name, max_pages)
        stop = time.time()
        print("La conversion to IMG [{0}] ha tardado "
              .format(pdf_file_name) + str(round(stop - start, 3)) + " segundos\n")

        if save_to_file:
            try:
                if not os.path.isdir(self.path_out):
                    os.mkdir(self.path_out)
                for img_idx, image_content in enumerate(content):
                    DocToImg.__save_img_file(self.path_out, pdf_file_name, image_content, (img_idx + 1))
            except OSError as e:
                self.logger.Error('DocToImg::run::{}'.format(e), sys.exc_info())
                return None

        return content
