import base.nsml_file_interface
import baseline.file_interface


class FileInterface(baseline.file_interface.FileInterface, base.nsml_file_interface.FileInterface):
    def __init__(self, glove_dir, glove_size, **kwargs):
        glove_dir = '/static/glove_squad'
        super(FileInterface, self).__init__(glove_dir, glove_size, **kwargs)
