import json
import os
import warnings

import httpx
import requests
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from Configuration import Configuration
from common.model.ClassFile import ClassFile
from commonsLib import loggerElk

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def get_category(gram_path):
    return ClassFile.get_containing_dir_name(gram_path)


class ClassifyService:
    # source type
    PLAINTEXT = "PLAINTEXT"
    FILE = "FILE"
    FOLDER = "FOLDER"
    VECTOR = "VECTOR"
    S3 = "S3"

    # mode type
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    GRAM = "GRAM"

    # action type
    PRE_PROCESS = "PRE_PROCESS"
    TRAIN = "TRAIN"
    PREDICT = "PREDICT"

    # execution mode
    NORMAL = "NORMAL"
    PARALLEL = "PARALLEL"

    # classification model type
    BAGGING = "BAGGING"
    BOOSTING_ADA = "BOOSTING_ADA"
    BOOSTING_SGD = "BOOSTING_SGD"
    DECISION_TREE = "DECISION_TREE"
    EXTRA_TREES = "EXTRA_TREES"
    NAIVE_BAYES_MULTI = "NAIVE_BAYES_MULTI"
    NAIVE_BAYES_COMPLEMENT = "NAIVE_BAYES_COMPLEMENT"
    RANDOM_FOREST = "RANDOM_FOREST"
    NN_NETWORK = "NN_NETWORK"
    ALL_VOTING = "ALL_VOTING"
    ALL_BY = "ALL_BY"

    class Service(object):
        TIMEOUT = 99999

        def __init__(self, conf: Configuration, lang: str = 'es'):  # TODO: Default lang ES
            self.conf = conf
            self.lang = lang
            self.SERVER_PRE_PROCESS_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_pp_host), int(self.conf.server_service_pp_port))
            self.SERVER_BAGGING_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_bag_host), int(self.conf.server_service_bag_port))
            self.SERVER_BOOSTING_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_bos_host), int(self.conf.server_service_bos_port))
            self.SERVER_DECISION_TREE_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_dt_host), int(self.conf.server_service_dt_port))
            self.SERVER_EXTRA_TREES_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_et_host), int(self.conf.server_service_et_port))
            self.SERVER_NAIVE_BAYES_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_nb_host), int(self.conf.server_service_nb_port))
            self.SERVER_RANDOM_FOREST_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_rf_host), int(self.conf.server_service_rf_port))
            self.SERVER_NN_NETWORK_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_nn_host), int(self.conf.server_service_nn_port))
            self.SERVER_VOTING_URL = "http://{0}:{1}/".format(
                str(self.conf.server_service_v_host), int(self.conf.server_service_v_port))
            self.logger = loggerElk(__name__, True)

        @staticmethod
        def local_server_up(server_url):
            response = requests.get(server_url)
            return response.status_code == 200

        @staticmethod
        def request(server_url,
                    source,
                    data='',
                    domain='',
                    file='',
                    model='',
                    mode='',
                    lang='',
                    force='',
                    bucket=''):
            body = {
                "source": source,
                "data": data,
                "domain": domain,
                "file": file,
                "model": model,
                "mode": mode,
                "lang": lang,
                "force": force,
                "bucket": bucket
            }
            payload = json.dumps(body)
            header = {"Content-Type": "application/json"}
            try:
                with httpx.Client() as client:
                    response = client.post(
                        server_url, data=payload, headers=header, timeout=ClassifyService.Service.TIMEOUT)
            except Exception as _:
                raise IOError

            return response

        @staticmethod
        async def request_async(client,
                                server_url,
                                source,
                                data='',
                                model='',
                                domain='',
                                file='',
                                mode='',
                                speed='',
                                lang=''):
            body = {
                "source": source,
                "mode": mode,
                "data": data,
                "domain": domain,
                "file": file,
                "model": model,
                "speed": speed,
                "lang": lang
            }
            payload = json.dumps(body)
            headers = {"Content-Type": "application/json"}
            return await client.post(
                url=server_url, data=payload, headers=headers, timeout=ClassifyService.Service.TIMEOUT)

    def __init__(self, conf):
        self.conf = conf
        self.service = self.Service(self.conf)
        self.logger = loggerElk(__name__, True)

    def get_server(self, model):
        server_url = ''
        if model == self.BAGGING:
            server_url = self.service.SERVER_BAGGING_URL
        elif model == self.BOOSTING_ADA or model == self.BOOSTING_SGD:
            server_url = self.service.SERVER_BOOSTING_URL
        elif model == self.DECISION_TREE:
            server_url = self.service.SERVER_DECISION_TREE_URL
        elif model == self.EXTRA_TREES:
            server_url = self.service.SERVER_EXTRA_TREES_URL
        elif model == self.NAIVE_BAYES_MULTI or model == self.NAIVE_BAYES_COMPLEMENT:
            server_url = self.service.SERVER_NAIVE_BAYES_URL
        elif model == self.RANDOM_FOREST:
            server_url = self.service.SERVER_RANDOM_FOREST_URL
        elif model == self.NN_NETWORK:
            server_url = self.service.SERVER_NN_NETWORK_URL
        elif model == self.ALL_VOTING:
            server_url = self.service.SERVER_VOTING_URL

        return server_url

    def pre_process(self, source, domain, mode, file=None, force=False, bucket=None):
        response = self.service.request(
            server_url=os.path.join(self.service.SERVER_PRE_PROCESS_URL, self.conf.server_pre_process),
            source=source,
            domain=domain,
            file=file,
            lang=self.service.lang,
            mode=mode,
            bucket=bucket,
            force=str(force))

        return response

    def pre_process_fit(self, source, domain, mode, force=False, bucket=None):
        response = self.service.request(
            server_url=os.path.join(self.service.SERVER_PRE_PROCESS_URL, self.conf.server_fit),
            source=source,
            domain=domain,
            lang=self.service.lang,
            mode=mode,
            bucket=bucket,
            force=str(force))

        return response

    def pre_process_transform(self, domain, mode, force=False, bucket=None):
        response = self.service.request(
            server_url=os.path.join(self.service.SERVER_PRE_PROCESS_URL, self.conf.server_fit),
            source=self.FOLDER,
            domain=domain,
            lang=self.service.lang,
            mode=mode,
            bucket=bucket,
            force=str(force))

        return response

    async def _train_async(self, source, client, data, domain, model, mode, execution=NORMAL):
        return await self.service.request_async(
            client=client,
            server_url=os.path.join(self.get_server(model), self.conf.server_training),
            source=source,
            data=data,
            domain=domain,
            model=model,
            mode=mode,
            speed=execution,
            lang=self.service.lang,
        )

    def _train(self, source, data, domain, model, mode, bucket=None):
        return self.service.request(
            server_url=os.path.join(self.get_server(model), self.conf.server_training),
            source=source,
            data=data,
            domain=domain,
            model=model,
            mode=mode,
            lang=self.service.lang,
            bucket=bucket
        )

    async def _predict_async(self, source, client, data, domain, file, model, mode):
        return await self.service.request_async(
            client=client,
            server_url=os.path.join(self.get_server(model), self.conf.server_predict),
            source=source,
            data=data,
            domain=domain,
            file=file,
            model=model,
            mode=mode,
            lang=self.service.lang,
        )

    def _predict(self, source, data, domain, file, model, mode, bucket=None):
        return self.service.request(
            server_url=os.path.join(self.get_server(model), self.conf.server_predict),
            source=source,
            data=data,
            domain=domain,
            file=file,
            model=model,
            mode=mode,
            lang=self.service.lang,
            bucket=bucket
        )

    @staticmethod
    def encode_categories(y):
        # encode class values as integers
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = to_categorical(encoded_y).astype(int)
        return dummy_y

    def show_metrics(self, y_test, y_predicted, stats=None):
        if y_test is None:
            self.logger.Information('Predicted: ' + str(y_predicted))
            stats.info = y_predicted
        elif y_predicted is not None and len(y_predicted) > 0:
            self.logger.Information('Accuracy: ' + str(accuracy_score(y_test, y_predicted[0])))
            # self.logger.Information(metrics.classification_report(y_test, y_predicted[0]))
            stats.info = metrics.classification_report(y_test, y_predicted[0])
        return stats

    def launch_bagging_classifier(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(' Classification by Bagging ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.BAGGING, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.BAGGING, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_bagging_classifier_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Bagging - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.BAGGING, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.BAGGING, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_boosting_ada(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(' Classification by Boosting ADA ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.BOOSTING_ADA, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.BOOSTING_ADA, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_boosting_ada_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Boosting ADA - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.BOOSTING_ADA, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.BOOSTING_ADA, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_boosting_sgd(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(' Classification by Boosting SGD ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.BOOSTING_SGD, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.BOOSTING_SGD, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_boosting_sgd_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Boosting SGD - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.BOOSTING_SGD, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.BOOSTING_SGD, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_decision_tree(self, source, data, domain, mode, action=None, file=None, bucket=None):
        self.logger.Information(' Classification by Decision Tree ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.DECISION_TREE, mode=mode,
                data=data, domain=domain, bucket=bucket)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.DECISION_TREE, mode=mode,
                data=data, domain=domain, file=file, bucket=bucket)

    async def launch_decision_tree_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Decision Tree - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.DECISION_TREE, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.DECISION_TREE, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_extra_trees(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(' Classification by Extra Trees ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.EXTRA_TREES, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.EXTRA_TREES, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_extra_trees_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Extra Trees - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.EXTRA_TREES, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.EXTRA_TREES, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_naive_bayes_complement(self, source, data, domain, mode, action=None, file=None, bucket=None):
        self.logger.Information(' Classification by Naive-Bayes Complement ')
        if not source:
            source = self.FOLDER
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.NAIVE_BAYES_COMPLEMENT, mode=mode,
                data=data, domain=domain, bucket=bucket)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.NAIVE_BAYES_COMPLEMENT, mode=mode,
                data=data, domain=domain, file=file, bucket=bucket)

    async def launch_naive_bayes_complement_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Naive-Bayes Complement - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.NAIVE_BAYES_COMPLEMENT, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.NAIVE_BAYES_COMPLEMENT, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_naive_bayes_multinomial(self, source, data, domain, mode, action=None, file=None, bucket=None):
        self.logger.Information(' Classification by Naive-Bayes Multinomial ')
        if not source:
            source = self.FOLDER
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.NAIVE_BAYES_MULTI, mode=mode,
                data=data, domain=domain, bucket=bucket)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.NAIVE_BAYES_MULTI, mode=mode,
                data=data, domain=domain, file=file, bucket=bucket)

    async def launch_naive_bayes_multinomial_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Naive-Bayes Multinomial - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.NAIVE_BAYES_MULTI, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.NAIVE_BAYES_MULTI, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_random_forest(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(' Classification by Random Forest ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.RANDOM_FOREST, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.RANDOM_FOREST, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_random_forest_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Random Forest - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.RANDOM_FOREST, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.RANDOM_FOREST, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_cnn_network(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(' Classification by Neural Network ')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.NN_NETWORK, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.NN_NETWORK, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_cnn_network_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(' Classification by Neural Network - asynchronous ')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.NN_NETWORK, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.NN_NETWORK, mode=mode,
                client=client, data=data, domain=domain, file=file)

    def launch_voting_classifier(self, source, data, domain, mode, action=None, file=None):
        self.logger.Information(f'Classification by Voting')
        if action == self.TRAIN:
            return self._train(
                source=source, model=self.ALL_VOTING, mode=mode,
                data=data, domain=domain)
        elif action == self.PREDICT:
            return self._predict(
                source=source, model=self.ALL_VOTING, mode=mode,
                data=data, domain=domain, file=file)

    async def launch_voting_classifier_async(self, source, data, domain, client, mode, action=None, file=None):
        self.logger.Information(f'Classification by Voting - asynchronous')
        if action == self.TRAIN:
            return await self._train_async(
                source=source, model=self.ALL_VOTING, mode=mode,
                client=client, data=data, domain=domain)
        elif action == self.PREDICT:
            return await self._predict_async(
                source=source, model=self.ALL_VOTING, mode=mode,
                client=client, data=data, domain=domain, file=file)
