import base64
import inspect
import os
import sys

import boto3
# start - JAEGER
import opentracing
from flask import request, jsonify
from flask_restplus import Resource, fields
from jaeger_client import Config
# end - JAEGER
from opentracing_utils import trace_requests

from Configuration import Configuration
from classifier.TextClass import TextClass
from common.model.SpacyModel import SpacyModel
from common.model.Stats import Stats
from commonsLib import loggerElk
from service.ClassifyFactory import ClassifyFactory


class GbcMlDocumentClassifierPrediction(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    predictionRequest = api.model('PredictionRequest', {
        'mode': fields.String(required=True, description='Source type (TEXT | IMAGE)'),
        'source': fields.String(required=True, description='Source of the data (PLAINTEXT | FOLDER | S3)'),
        'data': fields.String(required=True, description='Data resource identifier (url)'),
        'file': fields.String(required=True, description='Document to be predicted (file|data)'),
        'domain': fields.String(required=True, description='Domain resource identifier (domain name)'),
        'model': fields.String(required=True, description='Source of classifier '
                                                          '(BAGGING | BOOSTING_ADA | BOOSTING_SGD '
                                                          '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI '
                                                          '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST | ALL'
                                                          '| NN_NETWORK)'),
        'lang': fields.String(required=True, description='Language (es, en)')
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
    @api.expect(predictionRequest)
    def post(self):
        response = ''
        root_span = None
        try:
            stats = Stats()

            self.logger.Information('GbcMlDocumentClassifierPrediction::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            mode = request_payload['mode']
            data = request_payload['data']
            file = request_payload['file']
            domain = request_payload['domain']
            model = request_payload['model']

            # default params
            conf = Configuration(working_path=domain)
            _ = conf.dictionary  # TODO: Default
            lang = conf.lang  # TODO: Default
            # :domain: # TODO: It is local to the server working directory

            text_classifier = TextClass(conf)

            if source == ClassifyFactory.PLAINTEXT:

                if mode == ClassifyFactory.TEXT:

                    file_decoded = base64.b64decode(file).decode('utf-8')
                    try:
                        if not text_classifier.check_source(file_decoded):
                            self.logger.Error("Missing source files")
                        elif not text_classifier.check_model(model):
                            self.logger.Error("Missing model files")
                        elif not text_classifier.check_encoder():
                            self.logger.Error("Missing encoder file")
                        else:
                            nlp = SpacyModel.getInstance().getModel(lang)
                            ret = text_classifier.predict_by(
                                model=model, file_data=file_decoded, nlp=nlp)
                            response = stats.from_json(ret)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                else:
                    response = "Unsuitable type. Should be: TEXT"
                    self.logger.Debug(response)

            elif source == ClassifyFactory.FOLDER:

                if mode == ClassifyFactory.TEXT:
                    try:
                        if not text_classifier.check_source(data):
                            self.logger.Error("Missing source files")
                        elif not text_classifier.check_model(model):
                            self.logger.Error("Missing model files")
                        elif not text_classifier.check_encoder():
                            self.logger.Error("Missing encoder file")
                        else:
                            nlp = SpacyModel.getInstance().getModel(lang)
                            if not data:
                                ret = text_classifier.predict_by(
                                    model=model, file_url=file, nlp=nlp)
                            else:
                                ret = text_classifier.predict_by(
                                    model=model, data=data, nlp=nlp)
                            response = stats.from_json(ret)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                else:
                    response = "Unsuitable type. Should be: TEXT"
                    self.logger.Debug(response)

            elif source == ClassifyFactory.S3:
                _ = self.getS3Session()
                response = 'S3 not implemented'
                self.logger.Debug(response)

            else:
                response = "No valid source provided. Should be: PLAINTEXT | FOLDER | S3"
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

    @classmethod
    def getS3Session(cls):
        session = boto3.Session(
            aws_access_key_id=os.environ['ENV_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['ENV_SECRET_ACCESS_KEY']
        )
        s3 = session.client(u's3')
        return s3
