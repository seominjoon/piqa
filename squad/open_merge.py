import json
import os
import importlib
import sys
import numpy as np
import scipy

from pprint import pprint


# Predefined paths
PRED_PATH = './open_pred.json'


if __name__ == '__main__':
    from_ = importlib.import_module(sys.argv[1])
    FileInterface = from_.FileInterface
    ArgumentParser = from_.ArgumentParser

    # Parse arguments
    parser = ArgumentParser(description='Open setting evaluation')
    parser.add_arguments()
    args = parser.parse_args()
    args.pred_path = PRED_PATH

    ##### Copied from seve_demo (main.py) #####
    # Load model / processor / interface
    pprint(args.__dict__)
    interface = FileInterface(**args.__dict__)

    # Build index
    phrases = []
    paras = []
    results = []
    embs = []
    idxs = []
    iterator = interface.context_load(metadata=True, emb_type=args.emb_type)
    for (cur_phrases, each_emb, metadata) in iterator:
        embs.append(each_emb)
        phrases.extend(cur_phrases)
        for span in metadata['answer_spans']:
            results.append([len(paras), span[0], span[1]])
            idxs.append(len(idxs))
        paras.append(metadata['context'])

        if len(embs) == args.num_train_mats:
            break

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

        print('Index loaded: {}'.format(search_index.ntotal))

    elif args.emb_type == 'sparse':
        # from sklearn.neighbors import NearestNeighbors
        
        # IP will be used for search
        # TODO: memory issue
        raise NotImplementedError()
        search_index = scipy.sparse.vstack(embs).tocsr()
        # search_index = NearestNeighbors(n_neighbors=5, metric='l2', algorithm='brute').fit(embs_cat)

        def search(emb, k):
            scores = emb * search_index.T
            print(scores.shape)
            argmax_idx = np.argmax(scores, 1)
            print(argmax_idx.shape)
            return D[0], I[0]

        print('Index loaded: {}'.format(search_index.shape))

    else:
        raise ValueError()

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

        
    q_embs, q_ids = interface.question_load(emb_type=args.emb_type)
    q_embs = np.stack(q_embs, 0)
    print('Question loaded: {}'.format(q_embs.shape))
    outs = retrieve(q_embs, 1)
    print('Output retrieved: {}'.format(len(outs)))
    
    # Save dump
    predictions = {}
    assert len(q_ids) == len(outs)
    for q_id, q_out in zip(q_ids, outs):
        context, start, end, _ = q_out[0]
        predictions[q_id] = context[start:end]

    with open(args.pred_path, 'w') as fp:
        json.dump(predictions, fp)

    print('Prediction saved as {}'.format(args.pred_path))
