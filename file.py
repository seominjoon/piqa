import json
import os
import h5py


class FileInterface(object):
    def __init__(self, data_dir, glove_dir, save_dir, pred_path, question_emb_path):
        self.data_dir = data_dir
        self.glove_dir = glove_dir
        self._save_dir = save_dir
        self._pred_path = pred_path
        self._question_emb_path = question_emb_path
        self._question_emb = None
        self._save = None
        self._load = None

    def bind(self, save, load):
        self._save = save
        self._load = load

    def save(self, iteration):
        filename = os.path.join(self._save_dir, str(iteration))
        if not os.path.exists(filename):
            os.makedirs(filename)
        self._save(filename)

    def load(self, iteration, load_fn=None, session=None):
        if session is None:
            session = self._save_dir
        filename = os.path.join(session, str(iteration))
        if not os.path.exists(filename):
            os.makedirs(filename)
        self._load(filename)

    def pred(self, pred):
        with open(self._pred_path, 'w') as fp:
            json.dump(pred, fp)
            print('Prediction saved at %s' % self._pred_path)

    def report(self, **kwargs):
        print('Report something!')

    def question_emb(self, id_, emb):
        if self._question_emb is None:
            self._question_emb = h5py.File(self._question_emb_path, 'w')
        self._question_emb.create_dataset(id_, data=emb)
