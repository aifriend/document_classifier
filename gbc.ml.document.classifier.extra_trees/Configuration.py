import logging
import os

import yaml


class Configuration:

    def __init__(self, path="config.yml", working_path=''):
        try:
            with open(path, 'r') as yml_file:
                cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
        except Exception as e:
            logging.error('Loading configuration from {0} -> {1}'.format(path, str(e.args)))
            exit(1)

        development = cfg['development']

        # server
        server_service = development['service']
        server_service_url = server_service['url']
        self.server_training = server_service_url['training']
        self.server_predict = server_service_url['predict']
        server_service_address = server_service['address']
        self.server_host = server_service_address['host']
        self.server_port = server_service_address['port']

        # working directories
        directories = development['directories']
        self.dictionary = directories['dictionary']
        self.lang = directories['lang']
        self.working_path = working_path

        # pre_process
        pre_process = development['pre_process']
        self.pre_process_batch_size = pre_process['pre_process_batch_size']
        self.max_string_size = pre_process['max_string_size']

        # files
        files = development['files']
        self.tf = files['tf']
        self.tfidf = files['tfidf']
        self.vectorizer = files['vectorizer']
        self.vectorizer_type = files['vectorizer_tfidf']

        # system separator
        self.sep = os.path.sep

        # extra trees
        extra_trees = development['extra_trees']
        self.et_model = extra_trees['et_model']
        self.et_n_estimators = extra_trees['et_n_estimators']
        self.et_max_features = extra_trees['et_max_features']
        self.et_bootstrap = extra_trees['et_bootstrap']
        self.et_n_jobs = extra_trees['et_n_jobs']
        self.et_verbose = extra_trees['et_verbose']
