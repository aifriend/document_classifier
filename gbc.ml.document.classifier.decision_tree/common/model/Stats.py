import json
import sys

import numpy as np

from commonsLib import loggerElk


class StatsEncoder(json.JSONEncoder):
    def default(self, o):
        if "to_json" in dir(o):
            return o.to_json()


class Stats:

    def __init__(self,
                 classes: list = None, classifier: str = '',
                 info: str = '', predicted: list = None, history: list = None,
                 probabilities: list = None, result: str = 'OK'):
        self.classifier = classifier
        self.result = result
        self.info = info
        self.predicted = predicted
        self.classes = classes
        self.probabilities = probabilities
        self.history = history
        self.logger = self.logger = loggerElk(__name__, True)

    def update_response(self, response):
        if response is not None and len(response) == 3:
            self.predicted = response[0]
            self.classes = response[1]
            self.probabilities = response[2]

    def to_json(self):
        return {
            "classifier": self.classifier,
            "result": self.result,
            "info": self.info,
            "predicted": (self.predicted.tolist()
                          if isinstance(self.predicted, np.ndarray) else self.predicted),
            "classes": self.classes,
            "probabilities": self.probabilities,
            "history": self.history,
        }

    def from_json(self, json_object):
        try:
            return json.loads(json.dumps(json_object, indent=6, cls=StatsEncoder))
        except Exception as e:
            self.logger.Error('ERROR - Stats::from_json::{}'.format(e), sys.exc_info())
            return ""
