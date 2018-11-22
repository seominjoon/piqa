import baseline
import json


class FileInterface(baseline.FileInterface):
    def load_test(self):
        with open(self._test_path, 'r') as fp:
            squad = json.load(fp)
        return squad
