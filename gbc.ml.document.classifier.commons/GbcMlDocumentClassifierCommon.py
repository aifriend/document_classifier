import boto3
import inspect
# start - JAEGER
import opentracing
import os
import sys
from flask import request, jsonify
from flask_restplus import Resource, fields
from jaeger_client import Config
# end - JAEGER
from opentracing_utils import trace_requests

from commonsLib import loggerElk


class GbcMlDocumentClassifierCommon(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    predictionRequest = api.model('PredictionRequest', {
        'source': fields.String(required=True, description='Source of the data (PLAINTEXT | FILE | IMAGE | S3)'),
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
        root_span = None
        try:
            self.logger.Information('GbcMlDocumentClassifierPrediction::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            source = request_payload['source']

            if source == 'S3':
                _ = self.getS3Session()
            else:
                raise Exception('No valid source provided')

            res = {
                'result': 'ok',
            }

            return jsonify(res)

        except Exception as e:
            self.logger.Error('GbcMlDocumentClassifierPrediction::POST' + str(e.args), sys.exc_info())
            return {'message': 'Something went wrong: ' + str(e)}, 500

        finally:
            root_span.finish()

    @classmethod
    def getS3Session(cls):
        session = boto3.Session(
            aws_access_key_id=os.environ['ENV_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['ENV_SECRET_ACCESS_KEY']
        )
        s3 = session.client(u's3')
        return s3
