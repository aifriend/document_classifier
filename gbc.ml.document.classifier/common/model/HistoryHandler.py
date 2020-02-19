import json
import sys

from common.model.Stats import Stats
from commonsLib import loggerElk


class HistoryEncoder(json.JSONEncoder):
    @staticmethod
    def default(o, **kwargs):
        if "to_json" in dir(o):
            return o.to_json()


class HistoryHandler:
    DOC_IDX = "history"
    DOC_TYPE = "nn_image"

    def __init__(self, stats):
        self.stats = stats
        self.logger = loggerElk(__name__, True)

    def _history_to_string(self):
        history_item_list = dict()
        for k, v in self.stats.history.items():
            history_item = list()
            for item in v:
                history_item.append(str(item))
            history_item_list[k] = history_item
        return history_item_list

    def to_json(self):
        return {
            "history": self._history_to_string(),
        }

    def from_json(self, doc_content):
        try:
            doc_content = None
            if isinstance(doc_content, Stats) and doc_content.history:
                doc_content.history = str(doc_content.history)  # TODO: Introduce elastic-search

            # if doc_id is None:
            #     now = datetime.now()
            #     doc_id = datetime.timestamp(now)
            # log response to elasticsearch for future processing
            # logger.es.index(
            #     index=HistoryHandler.DOC_IDX, doc_type=HistoryHandler.DOC_TYPE, id=doc_id, body=content)

            return doc_content
        except Exception as e:
            self.logger.Error('HistoryHandler::from_json {}'.format(e), sys.exc_info())
            return doc_content
