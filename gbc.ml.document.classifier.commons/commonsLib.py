import certifi
import logging
import os
from datetime import datetime
from elasticsearch import Elasticsearch


class LogItem:

    def __init__(self, Message, Level, ObjectType, ObjectData):
        self.Message = Message
        self.Level = Level
        self.ObjectType = ObjectType
        self.ObjectData = ObjectData


class NoTrashFilter(logging.Filter):
    def __init__(self, loggingLevel):
        super().__init__()
        self.noTrashArray = ["_log", "LogInput", "LogResult", "Information", "Error", "Debug", "log_exception"]
        self.loggingLevel = loggingLevel

    def filter(self, record):
        return record.funcName in self.noTrashArray or record.levelno >= self.loggingLevel


class loggerFileAux:
    def __init__(self, debug_mode):
        self.DEBUG_MODE = debug_mode
        self.LOG_LIST = []

    def Log(self, level, message):
        if self.DEBUG_MODE:
            self.LOG_LIST.append({"level": level, "message": message})


class loggerElk:

    def __init__(self, owner__name__, enableKibana=False):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.lib_lob_level = str(os.environ["LIBRARIES_LOG_LEVEL"])
        switcher = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10
        }

        url = str(os.environ["ELK_URL"])

        self.elkIndex = os.environ["ELK_INDEX"]
        self.application = os.environ["APPLICATION"]
        self.environment = os.environ["ENVIRONMENT"]
        logLevel = os.environ["LOG_LEVEL"]
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
            logFile = os.environ["LOG_FILE"]
            fh = logging.FileHandler(logFile)
            fh.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
            fh.setLevel(logLevel)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except:
            self.logger.warning("LOG_FILE env-var not provided or can't write the file.")

        if enableKibana:
            self.elkEnabled = True
            try:
                self.es = Elasticsearch(str(url), use_ssl=True, ca_certs=certifi.where())
            except Exception as e:
                self.elkEnabled = False
                self.logger.error('CREATING CONNECTION TO ELK - ' + str(e))
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
            except:
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
