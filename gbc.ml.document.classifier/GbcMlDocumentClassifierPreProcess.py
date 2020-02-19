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


class GbcMlDocumentClassifierPreProcess(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    fitRequest = api.model('DocumentClassifierPreProcess', {
        'action': fields.String(required=True, description='Action to be executed (PRE_PROCESS, TRAIN, PREDICT)'),
        'source': fields.String(required=True, description='Source of the data (FILE | FOLDER | VECTOR)'),
        'domain': fields.String(required=True, description='Domain resource physical path'),
        'file': fields.String(required=False, description='Document to be predicted (file|data)'),
        'mode': fields.String(required=False, description='Source type (TEXT | IMAGE)'),
        'lang': fields.String(required=False, description='Language (es, en)'),
        'force': fields.String(required=False, description='Force pre-processing data (True/False)'),
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
        description='Document classifier pre-processing',
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

            self.logger.Information('GbcMlDocumentClassifierPreProcess::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            domain = request_payload['domain']
            file = request_payload['file']
            mode = request_payload['mode']
            force = request_payload['force']
            bucket = request_payload['bucket']

            image_classifier = ImageFactory()
            text_classifier = TextFactory()

            if source == ClassifyService.S3:

                if mode == ClassifyService.TEXT:
                    ret = text_classifier.pre_process(
                        source=source, domain=domain, mode=mode, bucket=bucket)
                    response = stats.from_json(ret)

                elif mode == ClassifyService.VECTOR:
                    ret = text_classifier.pre_process_vector(
                        source=source, domain=domain, mode=mode, force=force, bucket=bucket)
                    response = stats.from_json(ret)

                else:
                    response = "Unsuitable type. Should be: TEXT | VECTOR"
                    self.logger.Debug(response)

            elif source == ClassifyService.FILE:

                if mode == ClassifyService.TEXT or mode == ClassifyService.GRAM:
                    ret = text_classifier.pre_process(
                        source=source, domain=domain, mode=mode, file=file, force=force)
                    response = stats.from_json(ret)

                elif mode == ClassifyService.IMAGE:
                    ret = image_classifier.pre_process(
                        source=source, domain=domain, file=file, force=force)
                    response = stats.from_json(ret)

                else:
                    response = "Unsuitable type. Should be: TEXT | IMAGE"
                    self.logger.Debug(response)

            elif source == ClassifyService.FOLDER:

                if mode == ClassifyService.TEXT or mode == ClassifyService.GRAM:
                    ret = text_classifier.pre_process(
                        source=source, domain=domain, mode=mode, file=file, force=force)
                    response = stats.from_json(ret)

                elif mode == ClassifyService.IMAGE:
                    ret = image_classifier.pre_process(
                        source=source, domain=domain, file=file, force=force)
                    response = stats.from_json(ret)

                elif mode == ClassifyService.VECTOR:
                    ret = text_classifier.pre_process_vector(
                        source=source, domain=domain, force=force)
                    response = stats.from_json(ret)

                else:
                    response = "Unsuitable type. Should be: TEXT | IMAGE | VECTOR"
                    self.logger.Debug(response)

            else:
                response = "No valid source provided. Should be: VECTOR | FILE | FOLDER | S3"
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

