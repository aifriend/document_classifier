import inspect
import sys

# start - JAEGER
import opentracing
# end - JAEGER
from flask import jsonify, request
from flask_restplus import Resource, fields
from jaeger_client import Config
from opentracing_utils import trace_requests

from Configuration import Configuration
from classifier.TextClass import TextClass
from common.model.SpacyModel import SpacyModel
from common.s3.S3Service import S3Service
from commonsLib import loggerElk
from service.ClassifyFactory import ClassifyFactory


class GbcMlDocumentClassifierDataPreparationVectorizerResource(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    fitRequest = api.model('FitRequest', {
        'mode': fields.String(required=True, description='Source type (TEXT)'),
        'source': fields.String(required=True, description='Source of the data (FOLDER | S3)'),
        'domain': fields.String(required=True, description='Domain resource physical path'),
        'dictionary': fields.String(required=False, description='Dictionary to use (url)'),
        'lang': fields.String(required=False, description='Language (es, en)'),
        'bucket': fields.String(required=False, description='S3 Bucket'),
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
        description='Data Fit Vectorizer',
        responses={
            200: 'OK',
            400: 'Invalid Argument',
            500: 'Internal Error'})
    @api.expect(fitRequest)
    def post(self):
        response = ''
        root_span = None
        try:
            self.logger.Information('GbcMlDocumentClassifierDataPreparationVectorizerResource::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            mode = request_payload['mode']
            domain = request_payload['domain']
            bucket = request_payload['bucket']

            # default params
            conf = Configuration(working_path=domain)
            dictionary = conf.dictionary  # TODO: Default
            lang = conf.lang  # TODO: Default
            # :domain: # TODO: It is local to the server working directory

            text_classifier = TextClass(conf)

            if source == ClassifyFactory.FOLDER:

                if mode == ClassifyFactory.TEXT:

                    try:
                        if not text_classifier.check_txt():
                            response = "Provide required source files (ej.- *.pdf)"
                            self.logger.Debug(response)
                        else:
                            # Fit vectorizer
                            nlp = SpacyModel.getInstance().getModel(lang)
                            text_classifier.fit(nlp=nlp, dictionary=dictionary)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)

                else:
                    response = "Unsuitable type. Should be: TEXT"
                    self.logger.Debug(response)

            elif source == ClassifyFactory.S3:

                if mode == ClassifyFactory.VECTOR:
                    try:
                        s3_service = S3Service(bucket, domain)
                        documents = s3_service.get_files_from_s3()
                        if not text_classifier.s3_check_txt(documents):
                            response = f"No txt files found on bucket '{bucket}', domain '{domain}'"
                            self.logger.Debug(response)
                        else:
                            # Fit vectorizer
                            nlp = SpacyModel.getInstance().getModel(lang)
                            text_classifier.fit_s3(
                                nlp=nlp, dictionary=dictionary, s3_service=s3_service, s3_files=documents)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)

            else:
                response = "No valid source provided. Should be: FOLDER | S3"
                self.logger.Debug(response)

        except Exception as e:
            response = 'GbcMlDocumentClassifierDataPreparationVectorizerResource::POST' + str(e.args)
            self.logger.Error(response, sys.exc_info())
            return {'message': response}, 500

        finally:
            root_span.finish()
            res = {
                'result': 'ok',
                'response': response
            }
            return jsonify(res)
