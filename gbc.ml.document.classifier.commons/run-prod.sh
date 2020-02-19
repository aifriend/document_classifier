export FLASK_APP=api.py
export FLASK_DEBUG=1

# ENV variables
export ENV_ACCESS_KEY_ID=AKIAIQLDV23UPSQZC7QQ
export ENV_SECRET_ACCESS_KEY=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
export ENV_AWS_BUCKET=gbc.product.irph
export ELK_URL=https://search-samelan-elk-sandbox-4vyd2rkds6jljgamh7aofo6qam.eu-west-1.es.amazonaws.com
export LOG_FILE=logFile.log
export ELK_INDEX=gbcml-
export APPLICATION=GBC.ML.DOCUMENT.CLASSIFIER
export ENVIRONMENT=Development
export LOG_LEVEL=DEBUG
export LIBRARIES_LOG_LEVEL=ERROR
export JAEGER_AGENT_HOST=nexus.samelan.com
export JAEGER_AGENT_PORT=6831
# python -m flask run -h 0.0.0.0 -p 7116 --reload
gunicorn --bind 0.0.0.0:7116 wsgi:app
