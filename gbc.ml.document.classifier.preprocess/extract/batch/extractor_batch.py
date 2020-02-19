from Configuration import Configuration
from extract.GbcProcessAll import GbcProcessAll

if __name__ == '__main__':
    conf = Configuration(path="../../config.yml")

    # data_path = r"D:\Testing\class_services\data\raw"
    # GbcProcessAll.create_data(data_path)

    data_path = r"D:\Testing\class_services\data\test"
    gbc_process = GbcProcessAll()
    gbc_process.create_data_async(data_path)
