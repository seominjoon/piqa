import dev
import json


class FileInterface(dev.FileInterface):
    def __init__(self, **kwargs):
        self.analysis = kwargs['analysis']
        super(FileInterface, self).__init__(**kwargs)

    def load_test(self):
        with open(self._test_path, 'r') as fp:
            squad = json.load(fp)

        if self.analysis == 'eval':
            return squad
        elif self.analysis == 'open':
            test_examples = [{"idx": int("{}".format(idx)),
                              "cid": "Open_{}".format(idx),
                              "context": context} for (idx, context) in
                              enumerate(squad)]
            return test_examples
        else:
            raise ValueError()
