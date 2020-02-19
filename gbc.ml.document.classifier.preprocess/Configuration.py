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

        # data augmentation
        self.examples_per_case = development['augmentation']

        # server
        server_service = development['service']
        server_service_url = server_service['url']
        self.server_fit = server_service_url['fit']
        self.server_transform = server_service_url['transform']
        self.pdf_to_readable_url = server_service_url['pdf2readable']
        server_service_pp = server_service['pre_process']
        self.server_host = server_service_pp['host']
        self.server_port = server_service_pp['port']

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

        # pre process
        pre_process = development['pre_process']
        self.pre_process_batch_size = pre_process['pre_process_batch_size']
        self.max_string_size = pre_process['max_string_size']

        # files
        files = development['files']
        self.tf = files['tf']
        self.tfidf = files['tfidf']
        self.vectorizer = files['vectorizer']
        self.vectorizer_type = files['vectorizer_tfidf']
        self.image_file_ext = files['image_file_ext']
        self.text_file_ext = files['text_file_ext']
        self.source_file_ext = files['source_file_ext']
        self.gram_file_ext = files['gram_file_ext']

        # system separator
        self.sep = os.path.sep

        # threads
        thread_config = development['threads']
        self.max_threads_pdf_2_readable = thread_config['pdf2readable']
