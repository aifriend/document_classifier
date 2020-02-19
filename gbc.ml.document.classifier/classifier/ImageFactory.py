import asyncio
import os

import httpx
from PIL import Image

from Configuration import Configuration
from classifier.ClassifyService import ClassifyService
from classifier.Result import Result
from common.controller.ImageProcess import ImageProcess
from common.model.ClassFile import ClassFile
from common.model.Document import Document
from commonsLib import loggerElk


class ImageFactory:

    def __init__(self):
        self.conf = Configuration()
        self.from_image = ImageProcess()
        self.service = ClassifyService(self.conf)
        self.logger = loggerElk(__name__, True)

    def pre_process(self, source, domain, file, force):
        self.logger.Information('GbcMlDocumentClassifier::ImageFactory - pre-process data...')
        return self.service.pre_process(
            source=source, domain=domain, mode=self.service.IMAGE, file=file, force=force)

    async def _wait_for_task(self, process, source, data, domain, file=None):
        try:
            return await asyncio.wait_for(
                process(source, data, domain, file), timeout=self.service.Service.TIMEOUT)
        except asyncio.TimeoutError:
            self.logger.Information("GbcMlDocumentClassifier::ImageFactory - Timeout")

    async def _train_by_task_async(self, data, domain, file):
        self.logger.Information('GbcMlDocumentClassifier::ImageFactory - train by asynchronous...')
        async with httpx.AsyncClient() as client:
            tasks = list()
            tasks.append(self.service.launch_cnn_network_async(
                source=self.service.FOLDER,
                data=data, domain=domain, client=client, file=file,
                action=self.service.TRAIN, mode=self.service.IMAGE))

            response_list = list()
            for completed in asyncio.as_completed(tasks):
                ret = await completed
                stat = Result.update_response_async(ret)
                self.logger.Information(f"Response {stat.classifier}: Done!")
                response_list.append(stat)

            return response_list

    def train_all(self, model, data, domain):
        self.logger.Information(f'GbcMlDocumentClassifier::ImageFactory - training all models...')

        if model == self.service.ALL_BY:
            response = asyncio.run(self._wait_for_task(self._train_by_task_async, data, domain, None), debug=True)
        else:
            response = []

        return response

    def train_by(self, domain, model, data):
        stats = Result()
        self.logger.Information(f'GbcMlDocumentClassifier::ImageFactory - training model by...')
        stats.classifier = model

        if model == self.service.NN_NETWORK:
            response = self.service.launch_cnn_network(
                source=self.service.FOLDER,
                action=self.service.TRAIN, mode=self.service.IMAGE,
                data=data, domain=domain)
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be: NN_NETWORK'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    async def _predict_voting_task(self, source, data, domain, file):
        self.logger.Information('GbcMlDocumentClassifier::ImageFactory - predict voting task...')
        async with httpx.AsyncClient() as client:
            tasks = list()
            tasks.append(self.service.launch_voting_classifier_async(
                action=self.service.PREDICT, mode=self.service.IMAGE,
                source=source, data=data, domain=domain,
                file=file, client=client)
            )

        response_list = list()
        for completed in asyncio.as_completed(tasks):
            ret = await completed
            stat = Result.update_response_async(ret)
            self.logger.Information(f"Response {stat.classifier}: Done!")
            response_list.append(stat)

        return response_list

    def predict_all(self, source, data, domain, file):
        self.logger.Information(f'GbcMlDocumentClassifier::ImageFactory - training all models...')
        response = asyncio.run(self._wait_for_task(self._predict_voting_task, source, data, domain, file), debug=True)
        return Result.update_response_async(response)

    def predict_by(self, source, model, domain, data, file):
        stats = Result()
        self.logger.Information(f'GbcMlDocumentClassifier::ImageFactory - model prediction by...')
        stats.classifier = model

        if model == self.service.NN_NETWORK:
            response = self.service.launch_cnn_network(
                action=self.service.PREDICT, mode=self.service.IMAGE,
                source=source, data=data, domain=domain, file=file)
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be: NN_NETWORK'
            stats.result = 'WRONG_MODEL'

        stats.update_response(response)

        return stats

    def create_examples(self, path):
        i_list = ClassFile.list_files_ext(path, '.jpg')
        self.logger.Information(i_list)

        docs = []
        categories = set()

        for image in i_list:
            cropped = self.from_image.crop_image_loaded(
                self.from_image.resize_image_loaded(
                    self.from_image.load_image(image),
                    self.conf.resize_width,
                    self.conf.resize_height), self.conf.crop_width, self.conf.crop_height)
            name = ClassFile.get_file_name(image)
            categories.add(name[4:-3])
            ext = ClassFile.get_file_ext(image)
            # file = ClassFile.get_dir_name(image) + self.conf.sep + name + '_crop' + ext
            # Image.fromarray(cropped).save(file)
            examples = self.from_image.generate_examples(cropped, self.conf.examples_per_case)
            i = 0
            directory = os.path.join(path, name[4:-3])
            ClassFile.create_dir(directory)
            for example in examples:
                Image.fromarray(example).save(os.path.join(directory, name[4:] + '_' + str(i).zfill(3) + ext))
                docs.append(Document(example, name[4:-3]))
                i += 1

        ClassFile.list_to_file(categories, path)
