import os

os.environ['FLASK_APP'] = 'api.py'
os.environ['FLASK_DEBUG'] = '1'
os.environ['ENV_ACCESS_KEY_ID'] = 'AKIAIQLDV23UPSQZC7QQ'
os.environ['ENV_SECRET_ACCESS_KEY'] = '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'
os.environ['ENV_AWS_BUCKET'] = 'gbc.product.irph'
os.environ['ELK_URL'] = 'https://search-samelan-elk-sandbox-4vyd2rkds6jljgamh7aofo6qam.eu-west-1.es.amazonaws.com'
os.environ['LOG_FILE'] = 'logFile.log'
os.environ['ELK_INDEX'] = 'gbcml-'
os.environ['APPLICATION'] = 'GBC.ML.DOCUMENT.CLASSIFIER'
os.environ['ENVIRONMENT'] = 'Development'
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['LIBRARIES_LOG_LEVEL'] = 'ERROR'
os.environ['JAEGER_AGENT_HOST'] = 'nexus.samelan.com'
os.environ['JAEGER_AGENT_PORT'] = '6831'

from api import app

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=7116, debug=True)
