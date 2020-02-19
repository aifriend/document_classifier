#!/bin/bash
python3 -m venv pyvenv
. pyvenv/bin/activate
pip3 install -r ./requirements.txt
cp ./model/config_prod.yml ./model/config.yml
sh ./run-prod.sh
