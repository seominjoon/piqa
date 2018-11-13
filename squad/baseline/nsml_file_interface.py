import base.nsml_file_interface
import baseline.file_interface


class FileInterface(baseline.file_interface.FileInterface, base.nsml_file_interface.FileInterface):
    def __init__(self, glove_dir, elmo_options_file, elmo_weights_file, **kwargs):
        glove_dir = '/static/glove_squad'
        elmo_options_file = '/static/elmo/options.json'
        elmo_weights_file = '/static/elmo/weights.hdf5'
        super(FileInterface, self).__init__(glove_dir, elmo_options_file, elmo_weights_file, **kwargs)
