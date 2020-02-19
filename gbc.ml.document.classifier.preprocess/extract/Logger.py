import logging
from datetime import datetime

import certifi
from elasticsearch import Elasticsearch

from commonsLib import NoTrashFilter, LogItem


class Logger:

    def __init__(self, owner__name__, enableKibana=False):
        self.lib_lob_level = "ERROR"
        switcher = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10
        }

        url = "https://search-samelan-elk-sandbox-4vyd2rkds6jljgamh7aofo6qam.eu-west-1.es.amazonaws.com"

        self.elkIndex = "gbcml-"
        self.application = "GBC.ML.DOCUMENT.CLASSIFIER"
        self.environment = "Development"
        logLevel = "DEBUG"
        self.serviceName = str(owner__name__)
        logging.basicConfig(filemode='a')
        logging.getLogger().setLevel(logging.FATAL)
        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.logger.setLevel(logLevel)
        self.logger.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s (%(process)s %(threadName)s) - %(funcName)s -> %(lineno)s - %(message)s')

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
        ch.setLevel(logLevel)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        # create file handler which logs even debug messages
        try:
            logFile = "logFile.log"
            fh = logging.FileHandler(logFile)
            fh.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
            fh.setLevel(logLevel)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception as e:
            self.logger.warning("LOG_FILE env-var not provided or can't write the file::{}.".format(e))

        if enableKibana:
            self.elkEnabled = True
            try:
                self.es = Elasticsearch(str(url), use_ssl=True, ca_certs=certifi.where())
            except Exception as e:
                self.elkEnabled = False
                self.logger.debug('CREATING CONNECTION TO ELK - ' + str(e))
        else:
            self.elkEnabled = False

    def __sendItemToElk__(self, logItem: LogItem, extraAttrs=None):
        if self.elkEnabled:
            try:
                strDate = datetime.now().strftime("%Y.%m.%d")
                jsonBody = {
                    "file": self.serviceName,
                    "message": logItem.Message,
                    "@timestamp": datetime.now().isoformat(),
                    "level": logItem.Level,
                    "objectType": logItem.ObjectType,
                    "objectData": str(logItem.ObjectData),
                    "application": str(self.application),
                    "environment": str(self.environment)
                }
                if extraAttrs is not None:
                    for attribute, value in extraAttrs.items():
                        jsonBody[attribute] = value

                self.es.index(index=self.elkIndex + str(strDate), doc_type="TRACE",
                              body=jsonBody, params={"timeout": 0.0000000001})
            except Exception as e:
                self.logger.warning("{}.".format(e))
                pass

    def LogResult(self, message, ObjectData, extraAttrs=None):
        li = LogItem(message, 'Information', "result", ObjectData)
        self.logger.info(message + " - result - " + str(ObjectData))
        self.__sendItemToElk__(li, extraAttrs)

    def LogInput(self, message, ObjectData, extraAttrs=None):
        li = LogItem(message, 'Information', "input", ObjectData)
        self.logger.info(message + " - input - " + str(ObjectData))
        self.__sendItemToElk__(li, extraAttrs)

    def Information(self, message, extraAttrs=None):
        li = LogItem(message, 'Information', "trace", "")
        self.logger.info(message)
        self.__sendItemToElk__(li, extraAttrs)

    def Debug(self, message, extraAttrs=None):
        li = LogItem(message, 'Debug', "trace", "")
        self.logger.debug(message)
        self.__sendItemToElk__(li, extraAttrs)

    def Error(self, message, sysExecInfo=None):
        error = list()
        if sysExecInfo is not None:
            for e in sysExecInfo:
                if hasattr(e, 'tb_frame'):
                    error.append(str(e.tb_frame))
                else:
                    error.append(str(e))
        li = LogItem(message, 'Error', "trace", error)
        error.insert(0, message)

        self.logger.exception(str(error))
        self.__sendItemToElk__(li)
