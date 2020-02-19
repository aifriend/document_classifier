import base64
import inspect
import os
import sys

# start - JAEGER
import boto3
import opentracing
# end - JAEGER
from flask import jsonify, request
from flask_restplus import Resource, fields
from jaeger_client import Config
from opentracing_utils import trace_requests

from classifier.ClassifyService import ClassifyService
from classifier.ImageFactory import ImageFactory
from classifier.Result import Result
from classifier.TextFactory import TextFactory
from commonsLib import loggerElk


class GbcMlDocumentClassifierPredict(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    fitRequest = api.model('DocumentClassifierPredict', {
        'source': fields.String(required=True, description='Source of the data (FOLDER)'),
        'domain': fields.String(required=True, description='Domain resource physical path'),
        'data': fields.String(required=True, description='Data resource identifier (url)'),
        'file': fields.String(required=True, description='Document to be predicted (file|data)'),
        'model': fields.String(required=True, description='Model of classifier '
                                                          '(BAGGING | BOOSTING_ADA | BOOSTING_SGD '
                                                          '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI '
                                                          '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST | NN_NETWORK'
                                                          '| ALL_VOTING | ALL_BY'),
        'mode': fields.String(required=True, description='Source type (TEXT | IMAGE)'),
        'lang': fields.String(required=False, description='Language (es, en)'),
        'bucket': fields.String(required=False, description='Bucket for files in s3 source'),
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
        description='Document classifier predict',
        responses={
            200: 'OK',
            400: 'Invalid Argument',
            500: 'Internal Error'})
    @api.expect(fitRequest)
    def post(self):
        # model = ''
        # domain = ''
        response = ''
        root_span = None
        try:
            stats = Result()

            self.logger.Information('GbcMlDocumentClassifierPredict::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            model = request_payload['model']
            data = request_payload['data']
            file = request_payload['file']
            domain = request_payload['domain']
            mode = request_payload['mode']
            bucket = request_payload['bucket']

            image_classifier = ImageFactory()
            text_classifier = TextFactory()

            if source == ClassifyService.PLAINTEXT:

                if model == ClassifyService.ALL_VOTING or model == ClassifyService.ALL_BY:

                    if mode == ClassifyService.TEXT:
                        file_decoded = base64.b64decode(file).decode('utf-8')
                        ret = text_classifier.predict_all(
                            source=source, model=model, domain=domain, data=None, file=file_decoded)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT"
                        self.logger.Debug(response)

                else:

                    if mode == ClassifyService.TEXT:
                        file_decoded = base64.b64decode(file).decode('utf-8')
                        ret = text_classifier.predict_by(
                            source=source, model=model, domain=domain, data=None, file=file_decoded)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT"
                        self.logger.Debug(response)

            elif source == ClassifyService.FILE:

                if model == ClassifyService.ALL_VOTING or model == ClassifyService.ALL_BY:

                    if mode == ClassifyService.TEXT:
                        ret = text_classifier.predict_all(
                            source=source, model=model, domain=domain, data=None, file=file)
                        response = stats.from_json(ret)

                    elif mode == ClassifyService.IMAGE:
                        ret = image_classifier.predict_all(
                            source=source, domain=domain, data=None, file=file)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT | IMAGE"
                        self.logger.Debug(response)

                else:

                    if mode == ClassifyService.TEXT:
                        ret = text_classifier.predict_by(
                            source=source, model=model, domain=domain, data=None, file=file)
                        response = stats.from_json(ret)

                    elif mode == ClassifyService.IMAGE:
                        ret = image_classifier.predict_by(
                            source=source, model=model, domain=domain, data=None, file=file)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT | IMAGE"
                        self.logger.Debug(response)

            elif source == ClassifyService.FOLDER:

                if model == ClassifyService.ALL_VOTING or model == ClassifyService.ALL_BY:

                    if mode == ClassifyService.TEXT:
                        ret = text_classifier.predict_all(
                            source=source, model=model, domain=domain, data=data, file=None)
                        response = stats.from_json(ret)

                    elif mode == ClassifyService.IMAGE:
                        ret = image_classifier.predict_all(
                            source=source, domain=domain, data=data, file=None)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT | IMAGE"
                        self.logger.Debug(response)

                else:

                    if mode == ClassifyService.TEXT:
                        ret = text_classifier.predict_by(
                            source=source, model=model, domain=domain, data=data, file=None)
                        response = stats.from_json(ret)

                    elif mode == ClassifyService.IMAGE:
                        ret = image_classifier.predict_by(
                            source=source, model=model, domain=domain, data=data, file=None)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT | IMAGE"
                        self.logger.Debug(response)

            elif source == ClassifyService.S3:

                if mode == ClassifyService.TEXT:
                    ret = text_classifier.predict_by(source=source,
                                                     model=model, domain=domain, data=data, file=file,
                                                     bucket=bucket)
                    response = stats.from_json(ret)

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
            http_response = jsonify(res)
            # ClassFile.to_txtfile(res, f"{os.path.join(domain, model)}.json")  # TODO: Save prediction results to file
            return http_response

    @classmethod
    def getS3Session(cls):
        session = boto3.Session(
            aws_access_key_id=os.environ['ENV_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['ENV_SECRET_ACCESS_KEY']
        )
        s3 = session.client(u's3')
        return s3
