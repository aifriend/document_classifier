import json
import sys

from commonsLib import loggerElk


class StatsSparseVectorEncoder(json.JSONEncoder):

    def default(self, o):
        if "to_json" in dir(o):
            return o.to_json()
        return json.JSONEncoder.default(self, o)


class SparseVector:

    def __init__(self):
        self.data = []
        self._size = 0
        self.logger = loggerElk(__name__, True)

    def from_list(self, lst):
        self._size = len(lst)
        pairs = [(i, lst[i]) for i in range(len(lst))]
        # print(self.size)
        self.data = list(filter(lambda x: x[1] != 0, pairs))

    def to_json(self):
        return {
            "data": self.data,
        }

    def from_json(self, json_object):
        try:
            return json.loads(json.dumps(json_object, indent=1, cls=StatsSparseVectorEncoder))
        except Exception as e:
            self.logger.Error('SparseVector::from_json::{}'.format(e), sys.exc_info())
            return ''

    def __str__(self):
        return str(self.data)
