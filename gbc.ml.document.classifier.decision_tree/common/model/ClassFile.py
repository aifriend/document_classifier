import imghdr
import logging
import mimetypes
import os
import pickle
from pathlib import PurePath

import numpy as np
from binaryornot.check import is_binary


class ClassFile:

    @staticmethod
    def has_file(path, file):
        try:
            file_list = list()
            if os.path.isdir(path):
                file_list = ClassFile.list_files(path)
            elif os.path.isfile(path):
                file_list.append(path)
            if file_list and file:
                for file_item in file_list:
                    if file in file_item:
                        return True
            return False
        except Exception as ex:
            logging.exception(ex)
            return False

    @staticmethod
    def list_directory(path):
        """
        list all directories under specific route
        """
        files = []
        with os.scandir(path) as entries:
            for entry in entries:
                files.append(entry.name)

        return files

    @staticmethod
    def list_files(path):
        """
        list all files under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_pdf_files(path):
        """
        list all pdf files under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                file_root, file_extension = os.path.splitext(file)
                if 'pdf' in file_extension.lower():
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_files_ext(path, ext):
        """
        list all files under path with given extension
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                file_root, file_extension = os.path.splitext(file)
                if ext.lower() in file_extension.lower():
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def filter_files_root(file_list):
        file_filter = list()
        for file in file_list:
            file_root, file_extension = os.path.splitext(file)
            file_filter.append(file_root)

        return file_filter

    @staticmethod
    def filter_by_size(file_list):
        """
        filter all files with size greater than 0
        """
        file_filter = list()
        for file in file_list:
            if os.path.getsize(file) > 0:
                file_filter.append(file)

        return file_filter

    @staticmethod
    def filter_gram_duplicate(path, file_list):
        """
        filter all files with gram
        """
        filter_gram_list = ClassFile.filter_files_root(ClassFile.list_files_ext(path, "gram"))
        filter_list = file_list.copy()
        for file in file_list:
            file_root, _ = os.path.splitext(file)
            file_root, _ = os.path.splitext(file_root)
            if file_root in filter_gram_list:
                filter_list.remove(file)

        return filter_list

    @staticmethod
    def filter_duplicate(path, file_list, ext):
        """
        filter all files with ext
        """
        filter_by_ext = ClassFile.filter_files_root(
            ClassFile.filter_files_root(ClassFile.list_files_ext(path, ext)))
        filter_list = file_list.copy()
        for file in file_list:
            file_root, file_extension = os.path.splitext(file)
            if file_root in filter_by_ext:
                filter_list.remove(file)

        return filter_list

    @staticmethod
    def get_file_ext(file):
        """
        get the extension of a file
        """
        return os.path.splitext(file)[1]

    @staticmethod
    def get_dir_name(file):
        """
        get the the full path dir container of the file
        """
        return os.path.dirname(file)

    @staticmethod
    def get_containing_dir_name(file):
        """
        get just the name of the containing dir of the file
        """
        return PurePath(file).parent.name

    @staticmethod
    def file_base_name(file_name):
        """
        get the path and name of the file without the extension
        """
        if '.' in file_name:
            separator_index = file_name.index('.')
            base_name = file_name[:separator_index]
            return base_name
        else:
            return file_name

    @staticmethod
    def get_file_name(path):
        """
        get the name of the file without the extension
        """
        file_name = os.path.basename(path)
        return ClassFile.file_base_name(file_name)

    @staticmethod
    def create_dir(directory):
        """
        create a directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def list_to_file(set_, file_):
        """
        save a list to a file using pickle dump
        """
        with open(file_, 'wb') as fp:
            pickle.dump(sorted(list(set_)), fp)

    @staticmethod
    def to_txtfile(data, file_):
        """
        save data as a text file
        """
        with open(file_, "w") as output:
            output.write(str(data))

    @staticmethod
    def file_to_list(file_, binary=True):
        """
        read a list from a pickle file
        """
        list_ = []
        if os.path.getsize(file_) > 0:
            if binary:
                with open(file_, 'rb') as fp:
                    list_ = pickle.load(fp)
            else:
                with open(file_, 'r') as fp:
                    while True:
                        data = fp.readline()
                        if not data:
                            break
                        list_.append(data.splitlines())
        return sorted(list(list_))

    @staticmethod
    def csv_to_numpy_image(csv_file):
        """
        load numpy image from csv file
        """
        np.loadtxt(csv_file)

    @staticmethod
    def get_text(filename):
        """
        read from file as text
        """
        f = open(filename, "r", encoding="ISO-8859-1")
        return f.read()

    @staticmethod
    def save_sparse_csr(filename, mat):
        """
        save a sparse vector as a list of its elements
        """
        _ = mat.toarray().tofile(filename, sep=",", format="%f")

    @staticmethod
    def save_model(filename, model):
        """
        save scikit-learn model
        """
        pickle.dump(model, open(filename, 'wb'))

    @staticmethod
    def load_model(filename):
        """
        load scikit-learn model
        """
        try:
            if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                with open(filename, 'rb') as fp:
                    return pickle.load(fp)
        except Exception as e:
            logging.exception(e)
            return None

    @staticmethod
    def has_text_file(path):
        try:
            file_test_list = list()
            if os.path.isdir(path):
                file_test_list = ClassFile.list_files(path)
            elif os.path.isfile(path):
                file_test_list.append(path)
            for file_test in file_test_list:
                mime = mimetypes.guess_type(file_test)
                if mime and mime[0] == "text/plain":
                    return True
            return False
        except Exception as e:
            logging.exception(e)
            return False

    @staticmethod
    def has_gram_file(path):
        try:
            file_test_list = list()
            if os.path.isdir(path):
                file_test_list = ClassFile.list_files(path)
                file_test_list = file_test_list[-3]
            elif os.path.isfile(path):
                file_test_list.append(path)
            for file_test in file_test_list:
                if "gram" in file_test.lower():
                    return True
            return False
        except Exception as e:
            logging.exception(e)
            return False

    @staticmethod
    def has_media_file(path):
        try:
            file_test = None
            if os.path.isdir(path):
                file_test = ClassFile.list_files(path).pop()
            elif os.path.isfile(path):
                file_test = path
            if file_test is not None:
                return is_binary(file_test) and imghdr.what(file_test) in ["jpg", "jpeg", "png"]
            return False
        except Exception as e:
            logging.exception(e)
            return False
