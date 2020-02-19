import asyncio

import httpx

from Configuration import Configuration
from classifier.ClassifyService import ClassifyService
from classifier.Result import Result
from commonsLib import loggerElk


class TextFactory:

    def __init__(self):
        self.conf = Configuration()
        self.service = ClassifyService(self.conf)
        self.logger = loggerElk(__name__, True)

    def pre_process(self, source, domain, mode, bucket=None, file=None, force=False):
        self.logger.Information('GbcMlDocumentClassifier::TextFactory - pre-process source data...')
        ret = self.service.pre_process(
            source=source, domain=domain, mode=mode, file=file, force=force, bucket=bucket)
        return Result.get_response(ret)

    def pre_process_vector(self, source, domain, mode, force, bucket=None):
        self.logger.Information('GbcMlDocumentClassifier::TextFactory - pre-process data vectorizer...')
        ret = self.service.pre_process_fit(
            source=source, domain=domain, mode=mode, force=force, bucket=bucket)
        return Result.get_response(ret)

    async def _wait_for_task(self, process, source, data, domain, file=None):
        try:
            return await asyncio.wait_for(
                process(source, data, domain, file), timeout=self.service.Service.TIMEOUT)
        except asyncio.TimeoutError:
            self.logger.Information("GbcMlDocumentClassifier::TextFactory - Timeout")

    def _train_voting_task(self, data, domain):
        self.logger.Information('GbcMlDocumentClassifier::TextFactory - train voting task...')
        ret = self.service.launch_voting_classifier(
            source=self.service.FOLDER,
            data=data, domain=domain,
            action=self.service.TRAIN, mode=self.service.TEXT
        )
        return Result.update_response_async(ret)

    async def _train_by_task_async(self, data, domain, file):
        self.logger.Information('GbcMlDocumentClassifier::TextFactory - train by asynchronous...')
        async with httpx.AsyncClient() as client:
            tasks = list()
            tasks.append(asyncio.create_task(self.service.launch_decision_tree_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_extra_trees_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_naive_bayes_multinomial_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_naive_bayes_complement_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_random_forest_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_boosting_ada_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_boosting_sgd_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_bagging_classifier_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))

        response_list = list()
        for completed in asyncio.as_completed(tasks):
            ret = await completed
            stat = Result.update_response_async(ret)
            self.logger.Information(f"Response {stat.classifier}: Done!")
            response_list.append(stat)

        return response_list

    def train_all(self, model, data, domain):
        self.logger.Information(f'GbcMlDocumentClassifier::TextFactory - training all models...')

        if model == self.service.ALL_VOTING:
            response = self._train_voting_task(data, domain)
        elif model == self.service.ALL_BY:
            response = asyncio.run(
                self._wait_for_task(self._train_by_task_async, data, domain, None), debug=True)
        else:
            response = []

        return response

    def train_by(self, model, domain, data, bucket=None, source=None):
        stats = Result()
        self.logger.Information(f'GbcMlDocumentClassifier::TextFactory - training model by...')
        stats.classifier = model

        if model == self.service.BAGGING:
            response = self.service.launch_bagging_classifier(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT,
                source=source)
        elif model == self.service.BOOSTING_ADA:
            response = self.service.launch_boosting_ada(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT,
                source=source)
        elif model == self.service.BOOSTING_SGD:
            response = self.service.launch_boosting_sgd(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT,
                source=source)
        elif model == self.service.DECISION_TREE:
            response = self.service.launch_decision_tree(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT,  bucket=bucket,
                source=source)
        elif model == self.service.EXTRA_TREES:
            response = self.service.launch_extra_trees(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT,
                source=source)
        elif model == self.service.NAIVE_BAYES_MULTI:
            response = self.service.launch_naive_bayes_multinomial(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT, bucket=bucket,
                source=source)
        elif model == self.service.NAIVE_BAYES_COMPLEMENT:
            response = self.service.launch_naive_bayes_complement(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT, bucket=bucket,
                source=source)
        elif model == self.service.RANDOM_FOREST:
            response = self.service.launch_random_forest(
                data=data, domain=domain, action=self.service.TRAIN, mode=self.service.TEXT,
                source=source)
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be one of: BAGGING | BOOSTING_ADA | BOOSTING_SGD ' \
                         '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI ' \
                         '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    def _predict_voting_task(self, source, data, domain, file):
        self.logger.Information('GbcMlDocumentClassifier::TextFactory - predict voting task...')
        ret = self.service.launch_voting_classifier(
            source=source, data=data, domain=domain, file=file,
            action=self.service.PREDICT, mode=self.service.TEXT
        )
        return Result.update_response_async(ret)

    async def _predict_by_task_async(self, source, data, domain, file):
        response_list = list()

        self.logger.Information('GbcMlDocumentClassifier::TextFactory - predict by asynchronous...')

        async with httpx.AsyncClient() as client:
            tasks = list()
            tasks.append(asyncio.create_task(self.service.launch_decision_tree_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_extra_trees_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_naive_bayes_multinomial_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_random_forest_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_boosting_ada_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_bagging_classifier_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))

        for completed in asyncio.as_completed(tasks):
            ret = await completed
            stat = Result.update_response_async(ret)
            self.logger.Information(f"Response {stat.classifier}: Done!")
            response_list.append(stat)

        async with httpx.AsyncClient() as client:
            tasks = list()
            tasks.append(asyncio.create_task(self.service.launch_naive_bayes_complement_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))
            tasks.append(asyncio.create_task(self.service.launch_boosting_sgd_async(
                source=source, data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.TEXT,
            )))

        for completed in asyncio.as_completed(tasks):
            ret = await completed
            stat = Result.update_response_async(ret)
            self.logger.Information(f"Response {stat.classifier}: Done!")
            response_list.append(stat)

        return response_list

    def predict_all(self, source, model, data, domain, file):
        self.logger.Information(f'GbcMlDocumentClassifier::TextFactory - predict all models...')

        if model == self.service.ALL_VOTING:
            response = self._predict_voting_task(source, data, domain, file)
        elif model == self.service.ALL_BY:
            response = asyncio.run(self._wait_for_task(self._predict_by_task_async, source, data, domain, file),
                                   debug=True)
        else:
            response = []

        return response

    def predict_by(self, model, data, domain, file, bucket=None, source=None):
        stats = Result()
        self.logger.Information(f'GbcMlDocumentClassifier::TextFactory - model prediction by...')
        stats.classifier = model

        if model == self.service.BAGGING:
            response = self.service.launch_bagging_classifier(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT)
        elif model == self.service.BOOSTING_ADA:
            response = self.service.launch_boosting_ada(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT)
        elif model == self.service.BOOSTING_SGD:
            response = self.service.launch_boosting_sgd(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT)
        elif model == self.service.DECISION_TREE:
            response = self.service.launch_decision_tree(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT, bucket=bucket)
        elif model == self.service.EXTRA_TREES:
            response = self.service.launch_extra_trees(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT)
        elif model == self.service.NAIVE_BAYES_MULTI:
            response = self.service.launch_naive_bayes_multinomial(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT, bucket=bucket)
        elif model == self.service.NAIVE_BAYES_COMPLEMENT:
            response = self.service.launch_naive_bayes_complement(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT, bucket=bucket)
        elif model == self.service.RANDOM_FOREST:
            response = self.service.launch_random_forest(
                source=source, data=data, domain=domain, file=file,
                action=self.service.PREDICT, mode=self.service.TEXT)
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be one of: BAGGING | BOOSTING_ADA | BOOSTING_SGD ' \
                         '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI ' \
                         '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    @staticmethod
    def check_model(model=None):
        # is a available classification method
        if not (model == ClassifyService.BAGGING or model == ClassifyService.BOOSTING_ADA or
                model == ClassifyService.BOOSTING_SGD or model == ClassifyService.DECISION_TREE or
                model == ClassifyService.EXTRA_TREES or model == ClassifyService.NAIVE_BAYES_MULTI or
                model == ClassifyService.NAIVE_BAYES_COMPLEMENT or model == ClassifyService.RANDOM_FOREST):
            return False

        return True
