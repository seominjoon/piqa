from abc import ABCMeta

import torch.utils.data


class Processor(metaclass=ABCMeta):
    def construct(self, examples, metadata):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, in_):
        raise NotImplementedError()

    def preprocess(self, example):
        raise NotImplementedError()

    def postprocess(self, example, model_output):
        raise NotImplementedError()

    def postprocess_batch(self, dataset, model_input, model_output):
        raise NotImplementedError()

    def postprocess_context(self, example, context_output):
        raise NotImplementedError()

    def postprocess_context_batch(self, dataset, model_input, context_output):
        raise NotImplementedError()

    def postprocess_question(self, example, question_output):
        raise NotImplementedError()

    def postprocess_question_batch(self, dataset, model_input, question_output):
        raise NotImplementedError()

    def collate(self, examples):
        raise NotImplementedError()

    def process_metadata(self, metadata):
        raise NotImplementedError()

    def get_dump(self, dataset, input_, output, results):
        raise NotImplementedError()


class Sampler(torch.utils.data.Sampler, metaclass=ABCMeta):
    def __init__(self, dataset, data_type, **kwargs):
        self.dataset = dataset
        self.data_type = data_type
