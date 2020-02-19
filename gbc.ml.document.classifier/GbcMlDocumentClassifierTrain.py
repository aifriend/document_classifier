import inspect
import sys

# start - JAEGER
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


class GbcMlDocumentClassifierTrain(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    fitRequest = api.model('DocumentClassifierTrain', {
        'source': fields.String(required=True, description='Source of the data (FOLDER)'),
        'domain': fields.String(required=True, description='Domain resource physical path'),
        'data': fields.String(required=True, description='Data resource identifier (url)'),
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
        description='Document classifier train',
        responses={
            200: 'OK',
            400: 'Invalid Argument',
            500: 'Internal Error'})
    @api.expect(fitRequest)
    def post(self):
        response = ''
        root_span = None
        try:
            stats = Result()

            self.logger.Information('GbcMlDocumentClassifierTrain::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            model = request_payload['model']
            data = request_payload['data']
            domain = request_payload['domain']
            mode = request_payload['mode']
            bucket = request_payload['bucket']

            image_classifier = ImageFactory()
            text_classifier = TextFactory()

            if source == ClassifyService.FOLDER:

                if model == ClassifyService.ALL_VOTING or model == ClassifyService.ALL_BY:

                    if mode == ClassifyService.TEXT:
                        ret = text_classifier.train_all(
                            model=model, data=data, domain=domain)
                        response = stats.from_json(ret)

                    elif mode == ClassifyService.IMAGE:
                        ret = image_classifier.train_all(
                            model=model, data=data, domain=domain)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT | IMAGE"
                        self.logger.Debug(response)

                else:

                    if mode == ClassifyService.TEXT:
                        ret = text_classifier.train_by(
                            model=model, domain=domain, data=data, bucket=bucket)
                        response = stats.from_json(ret)

                    elif mode == ClassifyService.IMAGE:
                        ret = image_classifier.train_by(
                            model=model, domain=domain, data=data)
                        response = stats.from_json(ret)

                    else:
                        response = "Unsuitable type. Should be: TEXT | IMAGE"
                        self.logger.Debug(response)

            elif source == ClassifyService.S3:

                if mode == ClassifyService.TEXT:
                    ret = text_classifier.train_by(
                        model=model, domain=domain, data=data, bucket=bucket, source=source)
                    response = stats.from_json(ret)

                else:
                    response = "Unsuitable MODE type. Should be: TEXT "
                    self.logger.Error(response)

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
