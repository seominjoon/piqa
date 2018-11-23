import json
import os
import argparse
import dev
import analysis
import torch
import importlib
import sys
import numpy as np
import faiss

from pprint import pprint
from torch.utils.data import DataLoader


# Predefined paths
PRED_PATH = './open_pred.json'


# Customized argparser
class ArgumentParser(analysis.ArgumentParser):
    def __init__(self, description='baseline', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        super().add_arguments()

    def parse_args(self, **kwargs):
        args = super().parse_args()
        args.pred_path = PRED_PATH
        return args


if __name__ == '__main__':
    from_ = importlib.import_module(sys.argv[1])
    FileInterface = from_.FileInterface
    Processor = from_.Processor
    Sampler = from_.Sampler
    Model = from_.Model
    Loss = from_.Loss

    # Parse arguments
    parser = ArgumentParser(description='Open setting evaluation')
    parser.add_arguments()
    args = parser.parse_args()

    ##### Copied from seve_demo (main.py) #####
    # Load model / processor / interface
    device = torch.device('cuda' if args.cuda else 'cpu')
    pprint(args.__dict__)
    interface = FileInterface(**args.__dict__)

    # Build index
    with torch.no_grad():
        phrases = []
        paras = []
        results = []
        embs = []
        idxs = []
        iterator = interface.context_load(metadata=True, emb_type=args.emb_type)
        for _, (cur_phrases, each_emb, metadata) in zip(range(args.num_train_mats), iterator):
            embs.append(each_emb)
            phrases.extend(cur_phrases)
            for span in metadata['answer_spans']:
                results.append([len(paras), span[0], span[1]])
                idxs.append(len(idxs))
            paras.append(metadata['context'])
        if args.emb_type == 'dense':
            import faiss
            emb = np.concatenate(embs, 0)

            d = 4 * args.hidden_size * args.num_heads
            if args.metric == 'ip' or args.metric == 'cosine':
                quantizer = faiss.IndexFlatIP(d)  # Exact Search
            elif args.metric == 'l2':
                quantizer = faiss.IndexFlatL2(d)
            else:
                raise ValueError()

            if args.nlist != args.nprobe:
                # Approximate Search. nlist > nprobe makes it faster and less accurate
                if args.bpv is None:
                    if args.metric == 'ip':
                        search_index = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_INNER_PRODUCT)
                    elif args.metric == 'l2':
                        search_index = faiss.IndexIVFFlat(quantizer, d, args.nlist)
                    else:
                        raise ValueError()
                else:
                    assert args.metric == 'l2'  # only l2 is supported for product quantization
                    search_index = faiss.IndexIVFPQ(quantizer, d, args.nlist, args.bpv, 8)
                search_index.train(emb)
            else:
                search_index = quantizer

            search_index.add(emb)
            for cur_phrases, each_emb, metadata in iterator:
                phrases.extend(cur_phrases)
                for span in metadata['answer_spans']:
                    results.append([len(paras), span[0], span[1]])
                paras.append(metadata['context'])
                search_index.add(each_emb)

            if args.nlist != args.nprobe:
                search_index.nprobe = args.nprobe

            def search(emb, k):
                D, I = search_index.search(emb, k)
                return D, I
        else:
            raise NotImplementedError()

        def retrieve(q_embs, k):
            D, I = search(q_embs, k)
            outs = [[(paras[results[i][0]], results[i][1], 
                           results[i][2], '%.4r' % d.item(),)
                   for d, i in zip(D_k, I_k)] for D_k, I_k in zip(D, I)]
            return outs

        if args.mem_info:
            import psutil
            import os
            pid = os.getpid()
            py = psutil.Process(pid)
            info = py.memory_info()[0] / 2. ** 30
            print('Memory Use: %.2f GB' % info)

    print('Index loaded: {}'.format(search_index.ntotal))
        
    q_embs, q_ids = interface.question_load(emb_type=args.emb_type)
    q_embs = np.stack(q_embs, 0)
    print('Question loaded: {}'.format(q_embs.shape))
    outs = retrieve(q_embs, 1)
    print('Output retrieved: {}'.format(len(outs)))
    
    # Save dump
    predictions = {}
    for q_id, q_out in zip(q_ids, outs):
        context, start, end, _ = q_out[0]
        predictions[q_id] = context[start:end]

    with open(args.pred_path, 'w') as fp:
        json.dump(predictions, fp)

    print('Prediction saved as {}'.format(args.pred_path))
