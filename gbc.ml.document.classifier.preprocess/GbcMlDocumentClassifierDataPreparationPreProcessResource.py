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
from classifier.S3Class import S3Class
from classifier.TextClass import TextClass
from classifier.VisionClass import VisionClass
from common.model.SpacyModel import SpacyModel
from common.s3.S3Service import S3Service
from commonsLib import loggerElk
from extract.GbcProcessImage import GbcProcessImage
from extract.GbcProcessText import GbcProcessText
from service.ClassifyFactory import ClassifyFactory


class GbcMlDocumentClassifierDataPreparationPreProcessResource(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    fitRequest = api.model('PreProcessRequest', {
        'mode': fields.String(required=True, description='Source type (TEXT | IMAGE | GRAM)'),
        'source': fields.String(required=True, description='Source of the data (FILE | FOLDER | S3)'),
        'domain': fields.String(required=True, description='Domain resource physical path'),
        'file': fields.String(required=True, description='Document to be predicted (file|data)'),
        'dictionary': fields.String(required=False, description='Dictionary to use (url)'),
        'lang': fields.String(required=False, description='Language (es, en)'),
        'force': fields.String(required=False, description='Force process vectorizer'),
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
        description='Data Preparation',
        responses={
            200: 'OK',
            400: 'Invalid Argument',
            500: 'Internal Error'})
    @api.expect(fitRequest)
    def post(self):
        response = ''
        root_span = None
        try:
            self.logger.Information('GbcMlDocumentClassifierDataPreparationPreProcessResource::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']
            mode = request_payload['mode']
            domain = request_payload['domain']
            file = request_payload['file']
            force = request_payload['force']
            bucket = request_payload['bucket']

            # default params
            conf = Configuration(working_path=domain)
            dictionary = conf.dictionary  # TODO: Default
            lang = conf.lang  # TODO: Default
            # :domain: # TODO: It is local to the server working directory

            text_classifier = TextClass(conf)
            image_classifier = VisionClass(conf)
            s3_classifier = S3Class(conf)

            if source == ClassifyFactory.FOLDER:

                if mode == ClassifyFactory.GRAM:
                    try:
                        if force or not text_classifier.check_gram():
                            self.logger.Information("Source GRAM files: Trying to get GRAM from...")
                            nlp = SpacyModel.getInstance().getModel(lang)
                            text_classifier.gram(nlp=nlp, dictionary=dictionary)
                            if not text_classifier.check_gram():
                                response = "Provide required source files (ej.- *.pdf)"
                                self.logger.Debug(response)

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                elif mode == ClassifyFactory.TEXT:
                    try:
                        if force or not text_classifier.check_txt():
                            self.logger.Information("Source TEXT files: Trying to get TXT from...")
                            gbc_process = GbcProcessText(conf)
                            gbc_process.create_txt_async()
                            if not text_classifier.check_txt():
                                response = "Provide required source files (ej.- *.pdf)"
                                self.logger.Debug(response)

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                elif mode == ClassifyFactory.IMAGE:
                    try:
                        if force or not image_classifier.check_source():
                            self.logger.Information("Source IMAGE files: Trying to get JPG from...")
                            gbc_process = GbcProcessImage(conf)
                            gbc_process.create_data_async()
                            if not image_classifier.check_source():
                                response = "Provide required source files (ej.- *.pdf)"
                                self.logger.Debug(response)

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                else:
                    self.logger.Error("Unsuitable type. Should be: TEXT | IMAGE | GRAM")

            elif source == ClassifyFactory.FILE:

                if mode == ClassifyFactory.GRAM:
                    try:
                        if force or not text_classifier.check_file(file=file):
                            self.logger.Information("Source GRAM files: Trying to get GRAM from...")
                            gbc_process = GbcProcessText(conf)
                            gbc_process.process_file(file)
                            if not text_classifier.check_file(file=file):
                                response = "Provide required source files (ej.- *.pdf)"
                                self.logger.Debug(response)
                            else:
                                nlp = SpacyModel.getInstance().getModel(lang)
                                text_classifier.gram(nlp=nlp, dictionary=dictionary, file=file)

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                elif mode == ClassifyFactory.TEXT:
                    try:
                        if force or not text_classifier.check_file(file=file):
                            self.logger.Information("Source TEXT files. Trying to get TXT from...")
                            gbc_process = GbcProcessText(conf)
                            gbc_process.process_file(file)
                            if not text_classifier.check_file(file=file):
                                response = "Provide required source files (ej.- *.pdf)"
                                self.logger.Debug(response)

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                elif mode == ClassifyFactory.IMAGE:
                    try:
                        if force or not image_classifier.check_file(file=file):
                            self.logger.Information("Source IMAGE files: Trying to get JPG from...")
                            gbc_process = GbcProcessImage(conf)
                            gbc_process.process_file(file)
                            if not image_classifier.check_file(file=file):
                                response = "Provide required source files (ej.- *.pdf)"
                                self.logger.Debug(response)

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                else:
                    response = "Unsuitable type. Should be: TEXT | IMAGE | GRAM"
                    self.logger.Debug(response)

            elif source == ClassifyFactory.S3:

                if mode == ClassifyFactory.TEXT:
                    try:
                        s3_service = S3Service(bucket, domain)
                        documents = s3_service.get_files_from_s3()
                        if not text_classifier.s3_check_txt(documents):
                            s3_classifier.get_text_from_s3(
                                domain=domain, bucket=bucket, s3_session=s3_service.getS3Session())

                        response = f"{source} -> {str.upper(mode)} data has been pre-processed"
                        self.logger.Information(response)

                    except Exception as e:
                        response = 'Missing data!::{}'.format(e)
                        self.logger.Error(response)
                        return {'message': response}, 500

                else:
                    response = "Unsuitable type. Should be: TEXT"
                    self.logger.Debug(response)

            else:
                response = "No valid source provided. Should be: FILE | FOLDER | S3"
                self.logger.Debug(response)

        except Exception as e:
            response = 'GbcMlDocumentClassifierDataPreparationPreProcessResource::POST' + str(e.args)
            self.logger.Error(response, sys.exc_info())
            return {'message': response}, 500

        finally:
            root_span.finish()
            res = {
                'result': 'ok',
                'response': response
            }
            return jsonify(res)
