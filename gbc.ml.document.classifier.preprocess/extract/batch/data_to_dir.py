import logging
import os
import shutil

from Configuration import Configuration
from common.model.ClassFile import ClassFile

if __name__ == '__main__':
    conf = Configuration(path="../../config.yml")

    data_path = r"D:\Testing\data\raw"
    destination_path = r"D:\Testing\test\preprocess\text_data"

    logging.info("Batch file process::Moving data - From DATA to DIR")
    file_source_filter = ClassFile.list_files(data_path)
    filter_source_file = ClassFile.filter_by_ext(file_source_filter, conf.source_file_ext)
    filter_source_content_file = ClassFile.filter_by_size(filter_source_file)
    for file in filter_source_content_file:
        file_path, file_name = os.path.split(file)
        file_class = file_path.split("\\")[-1]
        file_destination = os.path.join(destination_path, file_class)
        if not os.path.isdir(file_destination):
            os.mkdir(file_destination)
        dest = shutil.copy(file, file_destination)
        print("Destination path: ", dest)
