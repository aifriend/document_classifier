import os
import pickle

from common.model.ClassFile import ClassFile
from commonsLib import loggerElk

logger = loggerElk(__name__, True)


class Singleton:
    # Here will be the instance stored.
    __conf = None
    __domain = None
    __instance = None
    __vectorizer = None

    def __init__(self, conf):
        """ Virtually private constructor. """
        if Singleton.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Singleton.__conf = conf
            Singleton.__domain = conf.working_path
            Singleton.__vectorizer = None

        Singleton.__instance = self

    @staticmethod
    def getInstance(conf):
        """ Static access method. """
        if Singleton.__instance is None:
            Singleton(conf)
        return Singleton.__instance

    @staticmethod
    def getFirstEncoder():
        if Singleton.__vectorizer is None:
            vectorizer_file_path = os.path.join(Singleton.__conf.working_path, Singleton.__domain)
            files = ClassFile.list_files_ext(vectorizer_file_path, Singleton.__conf.vectorizer_type)
            if files:
                for f in files:
                    # v_domain = Path(Singleton.__domain).stem
                    Singleton.__vectorizer = ClassFile.load_model(f)
                    break  # TODO: Just first vectorizer available

        return Singleton.__vectorizer

    @staticmethod
    def getFirstEncoderS3(s3Files, s3Service):
        if Singleton.__vectorizer is None:
            aux = Singleton.__conf.vectorizer_type.upper()
            files = [x for x in s3Files if x.Key.upper().endswith(f".{aux}")]
            if files:
                for f in files:
                    file_bytes = s3Service.get_byte_file(f.Key)
                    Singleton.__vectorizer = pickle.loads(file_bytes)
                    break  # TODO: Just first vectorizer availablez

        return Singleton.__vectorizer
