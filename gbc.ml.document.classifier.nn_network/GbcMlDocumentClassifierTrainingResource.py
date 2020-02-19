import inspect
import os
import sys

import boto3
# start - JAEGER
import opentracing
from flask import request, jsonify
from flask_restplus import Resource, fields
from jaeger_client import Config
from opentracing_utils import trace_requests

from Configuration import Configuration
from classifier.VisionClass import VisionClass
from common.model.HistoryHandler import HistoryHandler
from common.model.Stats import Stats
from commonsLib import loggerElk
# end - JAEGER
from service.AggregateFactory import AggregateFactory
from service.ClassifyFactory import ClassifyFactory

GLOBAL_DEBUG = True


class GbcMlDocumentClassifierTrainingResource(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    trainRequest = api.model('TrainRequest', {
        'source': fields.String(required=True, description='Source of the data (FOLDER | S3)'),
        'data': fields.String(required=True, description='Repo resource identifier (url)'),
        'domain': fields.String(required=True, description='Domain resource identifier (domain name)'),
        'model': fields.String(required=True, description='Source of classifier (NN_NETWORK)'),
        'mode': fields.String(required=True, description='Source type (IMAGE)'),
    })

    def __init__(self, *args, **kwargs):
        # start - JAEGER
        config = Config(config={'sampler': {'type': 'const', 'param': 1},
                                'logging': True
                                },
                        service_name=__name__)
        config.initialize_tracer()
        super().__init__(*args, **kwargs)

    trace_requests()  # noqa

    # end - JAEGER
    @api.doc(
        description='Process an IRPH tentative PDF',
        responses={
            200: 'OK',
            400: 'Invalid Argument',
            500: 'Internal Error'})
    @api.expect(trainRequest)
    def post(self):
        response = ''
        root_span = None
        try:
            stats = Stats()

            self.logger.Information('GbcMlDocumentClassifierTrainingResource::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            mode = request_payload['mode']
            data = request_payload['data']
            domain = request_payload['domain']
            model = request_payload['model']

            # default params
            conf = Configuration(working_path=domain)
            # :domain: # TODO: It is local to the server working directory

            vision_classifier = VisionClass(conf)

            if source == ClassifyFactory.FOLDER:

                if model == AggregateFactory.ALL_VOTING:

                    if mode == ClassifyFactory.IMAGE:
                        try:
                            if not vision_classifier.check_source(data):
                                self.logger.Debug("Missing source files!")
                            elif not vision_classifier.check_model(model):
                                self.logger.Debug("Missing model files!")
                            else:
                                ret = vision_classifier.train(
                                    model=model, data=data)
                                response = stats.from_json(ret)

                        except Exception as e:
                            response = 'Missing data!::{}'.format(e)
                            self.logger.Error(response)
                            return {'message': response}, 500

                    else:
                        response = "Unsuitable type. Should be: IMAGE"
                        self.logger.Debug(response)

                else:

                    if mode == ClassifyFactory.IMAGE:
                        try:
                            if not vision_classifier.check_source(data):
                                self.logger.Debug("Missing source files!")
                            elif not vision_classifier.check_model(model):
                                self.logger.Debug("Missing model files!")
                            else:
                                ret = vision_classifier.train_by(
                                    model=model, data=data)
                                history_handler = HistoryHandler(ret.history)
                                ret = history_handler.from_json(ret)
                                response = stats.from_json(ret)

                        except Exception as e:
                            response = 'Missing data!::{}'.format(e)
                            self.logger.Error(response)
                            return {'message': response}, 500

                    else:
                        response = "Unsuitable type. Should be: IMAGE"
                        self.logger.Debug(response)

            elif source == ClassifyFactory.S3:
                _ = self.getS3Session()
                response = 'S3 not implemented'
                self.logger.Debug(response)

            else:
                response = "No valid source provided. Should be: FOLDER | S3"
                self.logger.Debug(response)

        except Exception as e:
            response = 'GbcMlDocumentClassifierPrediction::POST' + str(e.args)
            self.logger.Error(response, sys.exc_info())
            return {'message': response}, 500

        finally:
            root_span.finish()
            res = {
                'result': 'ok',
                'response': response
            }
            return jsonify(res)

    class Student(object):
        def __init__(self, first_name: str, last_name: str):
            self.first_name = first_name
            self.last_name = last_name

    @classmethod
    def getS3Session(cls):
        session = boto3.Session(
            aws_access_key_id=os.environ['ENV_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['ENV_SECRET_ACCESS_KEY']
        )
        s3 = session.client(u's3')
        return s3
