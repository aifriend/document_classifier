import os

from Configuration import Configuration
from common.controller.TextProcess import TextProcess
from common.model.ClassFile import ClassFile
from common.model.SparseVector import SparseVector
from commonsLib import loggerElk


class TextClass:

    def __init__(self, conf: Configuration):
        self.conf = conf
        self.logger = loggerElk(__name__, True)

    def _initialize(self, nlp=None, dictionary=None):
        self.logger.Information('TextClass - loading configuration...')

        if dictionary is not None:
            self.conf.dictionary = dictionary
        if nlp is not None:
            self.logger.Information('TextClass - pre-processing...')
            self.pre_process = TextProcess(self.conf, nlp=nlp)

    def gram(self, nlp, dictionary, file=None):
        self._initialize(nlp=nlp, dictionary=dictionary)

        self.logger.Information('TextClass - get gram...')
        if file:
            return self.pre_process.pre_process_file(file)
        else:
            return self.pre_process.pre_process_batches()

    def fit(self, nlp, dictionary):
        self._initialize(nlp=nlp, dictionary=dictionary)

        self.logger.Information('TextClass - get vectorizer...')
        self.pre_process.pre_process_batches()
        self.pre_process.create_full_dataset_vectorizer()

    def fit_s3(self, nlp, dictionary, s3_service, s3_files=[]):
        self._initialize(nlp=nlp, dictionary=dictionary)

        self.logger.Information('TextClass - get vectorizer...')
        self.pre_process.pre_process_batches_s3(s3_service, s3_files)
        self.pre_process.create_full_dataset_vectorizer_s3(s3_service, s3_files)

    def transform(self, nlp, dictionary, data, file):
        self._initialize(nlp=nlp, dictionary=dictionary)

        vector = self.pre_process.transform(data=data, file=file)
        sparse_vector = SparseVector()
        sparse_vector.from_list([x for x in vector.toarray()[0]])
        self.logger.Information('TextClass - transform({0}) vectorizer: {1}'.format(file, vector))

        return sparse_vector

    def check_gram(self, data=None):
        # has right text type
        if data is None:
            if not ClassFile.has_gram_file(os.path.join(self.conf.working_path), depth=1):
                return False
        elif not ClassFile.has_gram_file(os.path.join(self.conf.working_path, data), depth=1):
            return False

        return True

    def check_txt(self, data=None):
        # has right text type
        if data is None:
            if not ClassFile.has_text_file(os.path.join(self.conf.working_path), depth=1):
                return False
        elif not ClassFile.has_text_file(os.path.join(self.conf.working_path, data), depth=1):
            return False

        return True

    @staticmethod
    def s3_check_txt(s3_elements):
        for obj in s3_elements:
            if obj.Size > 0 and obj.Key.upper().endswith('.TXT'):
                return True
        return False

    def check_file(self, file=None):
        # has right text type
        if file:
            file_root = ClassFile.get_file_root_name(file)
            file_like_list = ClassFile.list_files_like(path=self.conf.working_path, pattern=file_root)
            file_like_ext_list = ClassFile.filter_by_ext(file_like_list, ext=self.conf.text_file_ext)
            for file_guess in file_like_ext_list:
                if ClassFile.has_text_file(os.path.join(self.conf.working_path, file_guess), depth=1):
                    return True

        return False

    def check_encoder(self):
        # has right encoder
        if not ClassFile.list_files_ext(self.conf.working_path, self.conf.vectorizer_type):
            return False

        return True
