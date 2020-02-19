import logging

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

        # image processing
        image_processing = development['image_processing']
        self.resize_width = image_processing['resize_width']
        self.resize_height = image_processing['resize_height']
        self.crop_width = image_processing['crop_width']
        self.crop_height = image_processing['crop_height']

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

        # n_network
        n_network = development['nnetwork']
        self.nn_model_name = n_network['nn_model_name']
        self.nn_model_path = n_network['nn_model_path']
        self.nn_solver = n_network['nn_solver']
        self.nn_alpha = n_network['nn_alpha']
        self.nn_hidden_layer_sizes = n_network['nn_hidden_layer_sizes']
        self.nn_random_state = n_network['nn_random_state']
        self.nn_image_size = n_network['nn_image_size']
        self.nn_class_size = n_network['nn_class_size']
        self.nn_batch_size = n_network['nn_batch_size']
        self.nn_epochs = n_network['nn_epochs']
        self.nn_verbose = n_network['nn_verbose']
