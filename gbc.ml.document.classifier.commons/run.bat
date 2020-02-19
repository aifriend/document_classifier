set FLASK_APP=api.py
set FLASK_DEBUG=1

# ENV variables
set ENV_ACCESS_KEY_ID=AKIAIQLDV23UPSQZC7QQ
set ENV_SECRET_ACCESS_KEY=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
set ENV_AWS_BUCKET=gbc.product.irph
set ELK_URL=https://search-samelan-elk-sandbox-4vyd2rkds6jljgamh7aofo6qam.eu-west-1.es.amazonaws.com
set LOG_FILE=logFile.log
set ELK_INDEX=gbcml-
set APPLICATION=GBC.ML.DOCUMENT.CLASSIFIER
set ENVIRONMENT=Development
set LOG_LEVEL=DEBUG
set LIBRARIES_LOG_LEVEL=ERROR
set JAEGER_AGENT_HOST=nexus.samelan.com
set JAEGER_AGENT_PORT=6831

python -m flask run -h 0.0.0.0 -p 7116 --reload