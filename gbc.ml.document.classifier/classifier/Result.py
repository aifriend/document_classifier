import json
import sys

import numpy as np

from commonsLib import loggerElk


class StatsEncoder(json.JSONEncoder):
    def default(self, o):
        if "to_json" in dir(o):
            return o.to_json()


class Result:

    def __init__(self,
                 classifier: str = '', info: str = '', predicted: list = None,
                 history: list = None, probabilities: list = None,
                 result: str = 'OK', classes: list = None):
        self.classifier = classifier
        self.result = result
        self.info = info
        self.classes = classes
        self.predicted = predicted
        self.probabilities = probabilities
        self.history = history
        self.logger = self.logger = loggerElk(__name__, True)

    def _update_response(self, response):
        try:
            if response and isinstance(response, dict) and len(response) >= 7:
                self.classifier = response['classifier']
                self.result = response['result']
                self.info = response['info']
                self.classes = response['classes']
                self.predicted = response['predicted']
                self.probabilities = \
                    np.asarray(response['probabilities']).astype(np.float16).tolist()
                self.history = response['history']
        except Exception as _:
            self.logger.Error('Result::update_response', sys.exc_info())
            return ""

    @staticmethod
    def get_response(response):
        content_response = ''
        content = response.text
        if isinstance(content, str):
            json_content = json.loads(content)
            content_response = json_content['response']
        return content_response

    def update_response(self, response):
        content_response = self.get_response(response)
        self._update_response(content_response)

    def _update_response_async(self, response):
        try:
            content_response = self.get_response(response)
            if content_response:
                self._update_response(content_response)
        except Exception as _:
            self.logger.Error('Result::update_response', sys.exc_info())
            return ""

    @staticmethod
    def update_response_async(response):
        stat = Result()
        stat._update_response_async(response)
        return stat

    def to_json(self):
        return {
            "classifier": self.classifier,
            "result": self.result,
            "info": self.info,
            "classes": self.classes,
            "predicted": (self.predicted.tolist()
                          if isinstance(self.predicted, np.ndarray) else self.predicted),
            "probabilities": self.probabilities,
            "history": self.history,
        }

    def from_json(self, json_object):
        try:
            if json_object:
                if isinstance(json_object, list):
                    jo_list = list()
                    for jo in json_object:
                        jo_list.append(json.loads(json.dumps(jo, indent=6, cls=StatsEncoder)))
                    return jo_list
                else:
                    return json.loads(json.dumps(json_object, indent=7, cls=StatsEncoder))
        except Exception as _:
            self.logger.Error('Result::from_json', sys.exc_info())
            return ''
