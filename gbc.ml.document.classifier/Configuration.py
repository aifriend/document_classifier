import yaml


class Configuration:

    def __init__(self, conf_route="config.yml"):
        try:
            with open(conf_route, 'r') as yml_file:
                cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
        except Exception as e:
            print('ERROR - Configuration::__init__::{0}::{1}'.format(conf_route, str(e.args)))
            exit(1)

        development = cfg['development']

        # data augmentation
        self.examples_per_case = development['augmentation']

        # services
        server_service = development['service']
        server_service_url = server_service['url']
        self.server_fit = server_service_url['fit']
        self.server_transform = server_service_url['transform']
        self.server_pre_process = server_service_url['preprocess']
        self.server_training = server_service_url['training']
        self.server_predict = server_service_url['predict']
        # pre_process
        server_service_pp = server_service['pre_process']
        self.server_service_pp_host = server_service_pp['host']
        self.server_service_pp_port = server_service_pp['port']
        # bagging
        server_service_bag = server_service['bagging']
        self.server_service_bag_host = server_service_bag['host']
        self.server_service_bag_port = server_service_bag['port']
        # boosting
        server_service_bos = server_service['boosting']
        self.server_service_bos_host = server_service_bos['host']
        self.server_service_bos_port = server_service_bos['port']
        # decision tree
        server_service_dt = server_service['decision_tree']
        self.server_service_dt_host = server_service_dt['host']
        self.server_service_dt_port = server_service_dt['port']
        # extra_trees
        server_service_et = server_service['extra_trees']
        self.server_service_et_host = server_service_et['host']
        self.server_service_et_port = server_service_et['port']
        # naive_bayes
        server_service_nb = server_service['naive_bayes']
        self.server_service_nb_host = server_service_nb['host']
        self.server_service_nb_port = server_service_nb['port']
        # random_forest
        server_service_rf = server_service['random_forest']
        self.server_service_rf_host = server_service_rf['host']
        self.server_service_rf_port = server_service_rf['port']
        # nn_network
        server_service_nn = server_service['nn_network']
        self.server_service_nn_host = server_service_nn['host']
        self.server_service_nn_port = server_service_nn['port']
        # voting
        server_service_v = server_service['voting']
        self.server_service_v_host = server_service_v['host']
        self.server_service_v_port = server_service_v['port']

        # client
        client = development['client']
        self.client_host = client['host']
        self.client_port = client['port']
        client_service = client['service']
        self.client_pre_process = client_service['pre_process']
        self.client_training = client_service['training']
        self.client_predict = client_service['predict']

        # image processing
        image_processing = development['image_processing']
        self.resize_width = image_processing['resize_width']
        self.resize_height = image_processing['resize_height']
        self.crop_width = image_processing['crop_width']
        self.crop_height = image_processing['crop_height']
