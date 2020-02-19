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
        server = development['server']
        self.server_host = server['host']
        self.server_port = server['port']

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

        # system separator
        self.sep = os.path.sep

        # bagging
        bagging = development['bagging']
        self.bagging_model = bagging['bagging_model']
        self.bagging_n_estimators = bagging['bagging_n_estimators']
        self.bagging_max_samples = bagging['bagging_max_samples']
        self.bagging_n_jobs = bagging['bagging_n_jobs']
        self.bagging_verbose = bagging['bagging_verbose']

        # boosting
        boosting = development['boosting']
        self.boosting_model = boosting['boosting_model']
        self.boosting_n_estimators = boosting['boosting_n_estimators']
        self.boosting_verbose = boosting['boosting_verbose']

        # decision tree
        decision_tree = development['decision tree']
        self.dt_model = decision_tree['dt_model']
        self.dt_max_depth = decision_tree['dt_max_depth']
        self.dt_verbose = decision_tree['dt_verbose']

        # extra trees
        extra_trees = development['extra trees']
        self.et_model = extra_trees['et_model']
        self.et_n_estimators = extra_trees['et_n_estimators']
        self.et_max_features = extra_trees['et_max_features']
        self.et_bootstrap = extra_trees['et_bootstrap']
        self.et_n_jobs = extra_trees['et_n_jobs']
        self.et_verbose = extra_trees['et_verbose']

        # naive bayes
        naive_bayes = development['naive bayes']
        self.nb_model = naive_bayes['nb_model']
        self.nb_verbose = naive_bayes['nb_verbose']

        # random forest
        random_forest = development['random forest']
        self.rf_model = random_forest['rf_model']
        self.rf_n_estimators = random_forest['rf_n_estimators']
        self.rf_max_leaf_nodes = random_forest['rf_max_leaf_nodes']
        self.rf_n_jobs = random_forest['rf_n_jobs']
        self.rf_verbose = random_forest['rf_verbose']

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

        # voting
        voting = development['voting']
        self.voting_model = voting['voting_model']
        self.voting = voting['voting']
        self.voting_n_jobs = voting['voting_n_jobs']
        self.voting_verbose = voting['voting_verbose']
