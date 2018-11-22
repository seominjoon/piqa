import dev.nsml_file_interface
import json


class FileInterface(dev.nsml_file_interface.FileInterface):
    def load_test(self):
        with open(self._test_path, 'r') as fp:
            squad = json.load(fp)
        return squad
