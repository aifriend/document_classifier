from urllib.error import HTTPError

from flasgger import Swagger
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_restplus import Api

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
api = Api(app, version='1.0', prefix='/api', title='GBC Document IClassify API',
          description='Microservice to classify documents',
          )

# Enable Swagger and CORS
ns = api.namespace('gbc/ml/document/classifier',
                   description='Request Train/Predict document classification')
Swagger(app)
cors = CORS(app)

import sys

# JWT configuration
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
jwt = JWTManager(app)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']

from GbcMlDocumentClassifierDataPreparationVectorizerResource import \
    GbcMlDocumentClassifierDataPreparationVectorizerResource
from GbcMlDocumentClassifierDataPreparationPreProcessResource import \
    GbcMlDocumentClassifierDataPreparationPreProcessResource

ns.add_resource(GbcMlDocumentClassifierDataPreparationVectorizerResource, '/datapreparation/vectorizer')
ns.add_resource(GbcMlDocumentClassifierDataPreparationPreProcessResource, '/datapreparation/preprocess')


def service_avaliable():
    logger.LogResult("HealthCheck - OK", "service ok")
    return True, "service ok"


# HealthCheck
from healthcheck import HealthCheck, EnvironmentDump

envdump = EnvironmentDump()
health = HealthCheck(checkers=[service_avaliable])
app.add_url_rule("/healthcheck", "healthcheck", view_func=lambda: health.check())

from commonsLib import loggerElk

logger = loggerElk(__name__, True)


@api.errorhandler(Exception)
def handle_error(e):
    _logger = loggerElk(__name__)
    _logger.Information("Error Handler")
    code = 500
    if isinstance(e, HTTPError):
        code = e.code
    logger.Error(str(e), sys.exc_info())
    return {'message': 'Something went wrong: ' + str(e)}, code
