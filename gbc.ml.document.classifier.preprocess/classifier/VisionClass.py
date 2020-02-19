import os

from Configuration import Configuration
from common.model.ClassFile import ClassFile
from commonsLib import loggerElk


class VisionClass:

    def __init__(self, conf: Configuration):
        self.conf = conf
        self.logger = loggerElk(__name__, True)

    def _initialize(self):
        self.logger.Information('VisionClass - loading configuration...')

    def check_source(self, data=None):
        # has right text type
        if data is None:
            if not ClassFile.has_media_file(os.path.join(self.conf.working_path), depth=1):
                return False
        elif not ClassFile.has_media_file(os.path.join(self.conf.working_path, data), depth=1):
            return False

        return True

    def check_file(self, file=None):
        # has right text type
        if file:
            file_root = ClassFile.get_file_root_name(file)
            file_like_list = ClassFile.list_files_like(path=self.conf.working_path, pattern=file_root)
            file_like_ext_list = ClassFile.filter_by_ext(file_like_list, ext=self.conf.image_file_ext)
            for file_guess in file_like_ext_list:
                if ClassFile.has_media_file(os.path.join(self.conf.working_path, file_guess), depth=1):
                    return True

        return False
