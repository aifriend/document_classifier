import en_core_web_md
import es_core_news_md

from common.model.ClassFile import ClassFile
from commonsLib import loggerElk

logger = loggerElk(__name__, True)


class SpacyModel:
    __instance = None
    model = dict()
    dictionary = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if SpacyModel.__instance is None:
            SpacyModel()
        return SpacyModel.__instance

    def getModel(self, lang):
        return self.model[lang]

    def getDictionary(self, dictionary_path):
        if not self.dictionary:
            logger.Information('SpacyModel - loading spacy\'s dictionary...')
            dict_file = ClassFile.get_text(dictionary_path).split('\n')
            dict_file = filter(lambda x: (x is not ''), dict_file)
            spa_dict = list(map(lambda x: x.lower(), dict_file))
            self.dictionary = spa_dict

        return self.dictionary

    def __init__(self):
        """ Virtually private constructor. """
        if SpacyModel.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            logger.Information('SpacyModel - loading spacy\'s model...')
            self.model['es'] = es_core_news_md.load()
            self.model['en'] = en_core_web_md.load()
            SpacyModel.__instance = self
