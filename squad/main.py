import sys
import time
from collections import OrderedDict
from pprint import pprint
import importlib

from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import torch
import numpy as np
from torch.utils.data import DataLoader

import base


def preprocess(interface, args):
    """Helper function for caching preprocessed data
    """
    print('Loading train and dev data')
    train_examples = interface.load_train()
    dev_examples = interface.load_test()

    # load metadata, such as GloVe
    print('Loading metadata')
    metadata = interface.load_metadata()

    print('Constructing processor')
    processor = Processor(**args.__dict__)
    processor.construct(train_examples, metadata)

    # data loader
    print('Preprocessing datasets and metadata')
    train_dataset = tuple(processor.preprocess(example) for example in train_examples)
    dev_dataset = tuple(processor.preprocess(example) for example in dev_examples)
    processed_metadata = processor.process_metadata(metadata)

    print('Creating data loaders')
    train_sampler = Sampler(train_dataset, 'train', **args.__dict__)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=processor.collate, sampler=train_sampler)

    dev_sampler = Sampler(dev_dataset, 'dev', **args.__dict__)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                            collate_fn=processor.collate, sampler=dev_sampler)

    if args.preload:
        train_loader = tuple(train_loader)
        dev_loader = tuple(dev_loader)

    out = {'processor': processor,
           'train_dataset': train_dataset,
           'dev_dataset': dev_dataset,
           'processed_metadata': processed_metadata,
           'train_loader': train_loader,
           'dev_loader': dev_loader}

    return out


def train(args):
    start_time = time.time()
    device = torch.device('cuda' if args.cuda else 'cpu')

    pprint(args.__dict__)
    interface = FileInterface(**args.__dict__)
    out = interface.cache(preprocess, args) if args.cache else preprocess(interface, args)
    processor = out['processor']
    processed_metadata = out['processed_metadata']
    train_dataset = out['train_dataset']
    dev_dataset = out['dev_dataset']
    train_loader = out['train_loader']
    dev_loader = out['dev_loader']

    model = Model(**args.__dict__).to(device)
    model.init(processed_metadata)

    loss_model = Loss(**args.__dict__).to(device)
    optimizer = torch.optim.Adam(p for p in model.parameters() if p.requires_grad)

    interface.bind(processor, model, optimizer=optimizer)

    step = 0
    train_report, dev_report = None, None

    print('Training')
    interface.save_args(args.__dict__)
    model.train()
    for epoch_idx in range(args.epochs):
        for i, train_batch in enumerate(train_loader):
            train_batch = {key: val.to(device) for key, val in train_batch.items()}
            model_output = model(step=step, **train_batch)
            train_results = processor.postprocess_batch(train_dataset, train_batch, model_output)
            train_loss = loss_model(step=step, **model_output, **train_batch)
            train_f1 = float(np.mean([result['f1'] for result in train_results]))
            train_em = float(np.mean([result['em'] for result in train_results]))

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1

            # report & eval & save
            if step % args.report_period == 1:
                train_report = OrderedDict(step=step, train_loss=train_loss.item(), train_f1=train_f1,
                                           train_em=train_em, time=time.time() - start_time)
                print(interface.report(**train_report))

            if step % args.eval_save_period == 1:
                with torch.no_grad():
                    model.eval()
                    loss_model.eval()
                    pred = {}
                    dev_losses, dev_results = [], []
                    for dev_batch, _ in zip(dev_loader, range(args.eval_steps)):
                        dev_batch = {key: val.to(device) for key, val in dev_batch.items()}
                        model_output = model(**dev_batch)
                        results = processor.postprocess_batch(dev_dataset, dev_batch, model_output)

                        dev_loss = loss_model(step=step, **dev_batch, **model_output)

                        for result in results:
                            pred[result['id']] = result['pred']
                        dev_results.extend(results)
                        dev_losses.append(dev_loss.item())

                    dev_loss = float(np.mean(dev_losses))
                    dev_f1 = float(np.mean([result['f1'] for result in dev_results]))
                    dev_em = float(np.mean([result['em'] for result in dev_results]))
                    dev_f1_best = dev_f1 if dev_report is None else max(dev_f1, dev_report['dev_f1_best'])
                    dev_f1_best_step = step if dev_report is None or dev_f1 > dev_report['dev_f1_best'] else dev_report[
                        'dev_f1_best_step']

                    dev_report = OrderedDict(step=step, dev_loss=dev_loss, dev_f1=dev_f1, dev_em=dev_em,
                                             time=time.time() - start_time, dev_f1_best=dev_f1_best,
                                             dev_f1_best_step=dev_f1_best_step)

                    summary = False
                    if dev_report['dev_f1_best_step'] == step:
                        summary = True
                        interface.save(iteration=step)
                        interface.pred(pred)
                    print(interface.report(summary=summary, **dev_report))
                    model.train()
                    loss_model.train()

            if step == args.train_steps:
                break
        if step == args.train_steps:
            break


def test(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    pprint(args.__dict__)

    interface = FileInterface(**args.__dict__)
    # use cache for metadata
    if args.cache:
        out = interface.cache(preprocess, args)
        processor = out['processor']
        processed_metadata = out['processed_metadata']
    else:
        processor = Processor(**args.__dict__)
        metadata = interface.load_metadata()
        processed_metadata = processor.process_metadata(metadata)

    model = Model(**args.__dict__).to(device)
    model.init(processed_metadata)
    interface.bind(processor, model)

    interface.load(args.iteration, session=args.load_dir)

    test_examples = interface.load_test()
    test_dataset = tuple(processor.preprocess(example) for example in test_examples)

    test_sampler = Sampler(test_dataset, 'test', **args.__dict__)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             collate_fn=processor.collate)

    print('Inferencing')
    with torch.no_grad():
        model.eval()
        pred = {}
        for batch_idx, (test_batch, _) in enumerate(zip(test_loader, range(args.eval_steps))):
            test_batch = {key: val.to(device) for key, val in test_batch.items()}
            model_output = model(**test_batch)
            results = processor.postprocess_batch(test_dataset, test_batch, model_output)
            if batch_idx % args.dump_period == 0:
                dump = processor.get_dump(test_dataset, test_batch, model_output, results)
                interface.dump(batch_idx, dump)
            for result in results:
                pred[result['id']] = result['pred']

            print('[%d/%d]' % (batch_idx + 1, len(test_loader)))
        interface.pred(pred)


def embed(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    pprint(args.__dict__)

    interface = FileInterface(**args.__dict__)
    # use cache for metadata
    if args.cache:
        out = interface.cache(preprocess, args)
        processor = out['processor']
        processed_metadata = out['processed_metadata']
    else:
        processor = Processor(**args.__dict__)
        metadata = interface.load_metadata()
        processed_metadata = processor.process_metadata(metadata)

    model = Model(**args.__dict__).to(device)
    model.init(processed_metadata)
    interface.bind(processor, model)

    interface.load(args.iteration, session=args.load_dir)

    test_examples = interface.load_test()
    test_dataset = tuple(processor.preprocess(example) for example in test_examples)

    test_sampler = Sampler(test_dataset, 'test', **args.__dict__)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             collate_fn=processor.collate)

    print('Saving embeddings')
    with torch.no_grad():
        model.eval()
        for batch_idx, (test_batch, _) in enumerate(zip(test_loader, range(args.eval_steps))):
            test_batch = {key: val.to(device) for key, val in test_batch.items()}

            if args.mode == 'embed' or args.mode == 'embed_context':

                context_output = model.get_context(**test_batch)
                context_results = processor.postprocess_context_batch(test_dataset, test_batch, context_output)

                for id_, phrases, matrix, metadata in context_results:
                    if not args.metadata:
                        metadata = None
                    interface.context_emb(id_, phrases, matrix, metadata=metadata, emb_type=args.emb_type)

            if args.mode == 'embed' or args.mode == 'embed_question':

                question_output = model.get_question(**test_batch)
                question_results = processor.postprocess_question_batch(test_dataset, test_batch, question_output)

                for id_, emb in question_results:
                    interface.question_emb(id_, emb, emb_type=args.emb_type)

            print('[%d/%d]' % (batch_idx + 1, len(test_loader)))

    if args.archive:
        print('Archiving')
        interface.archive()


def serve(args):
    # serve_demo: Load saved embeddings, serve question model. question in, results out.
    # serve_question: only serve question model. question in, vector out.
    # serve_context: only serve context model. context in, phrase-vector pairs out.
    # serve: serve all three.
    device = torch.device('cuda' if args.cuda else 'cpu')
    pprint(args.__dict__)

    interface = FileInterface(**args.__dict__)
    # use cache for metadata
    if args.cache:
        out = interface.cache(preprocess, args)
        processor = out['processor']
        processed_metadata = out['processed_metadata']
    else:
        processor = Processor(**args.__dict__)
        metadata = interface.load_metadata()
        processed_metadata = processor.process_metadata(metadata)

    model = Model(**args.__dict__).to(device)
    model.init(processed_metadata)
    interface.bind(processor, model)

    interface.load(args.iteration, session=args.load_dir)

    with torch.no_grad():
        model.eval()

        if args.mode == 'serve_demo':
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
                    return D[0], I[0]

            elif args.emb_type == 'sparse':
                assert args.metric == 'l2'  # currently only l2 exact search is supported

                embs_cat = scipy.sparse.vstack(embs).tocsr()
                search_index = NearestNeighbors(n_neighbors=5, metric='l2', algorithm='brute').fit(embs_cat)

                for cur_phrases, each_emb, metadata in iterator:
                    raise Exception()
                    phrases.extend(cur_phrases)
                    for span in metadata['answer_spans']:
                        results.append([len(paras), span[0], span[1]])
                    paras.append(metadata['context'])
                    search_index.addDataPointBatch(each_emb.tocsr())

                def search(emb, k):
                    emb = emb.tocsr()
                    nbrs = search_index.kneighbors(emb)
                    D, I = nbrs
                    return D[0], I[0]

            else:
                raise ValueError()

            def retrieve(question, k):
                example = {'question': question, 'id': 'real', 'idx': 0}
                dataset = (processor.preprocess(example),)
                loader = DataLoader(dataset, batch_size=1, collate_fn=processor.collate)
                batch = next(iter(loader))
                question_output = model.get_question(**batch)
                question_results = processor.postprocess_question_batch(dataset, batch, question_output)
                id_, emb = question_results[0]
                D, I = search(emb, k)
                out = [(paras[results[i][0]], results[i][1], results[i][2], '%.4r' % d.item(),)
                       for d, i in zip(D, I)]
                return out

            if args.mem_info:
                import psutil
                import os
                pid = os.getpid()
                py = psutil.Process(pid)
                info = py.memory_info()[0] / 2. ** 30
                print('Memory Use: %.2f GB' % info)

            # Demo server. Requires flask and tornado
            from flask import Flask, request, jsonify
            from flask_cors import CORS

            from tornado.wsgi import WSGIContainer
            from tornado.httpserver import HTTPServer
            from tornado.ioloop import IOLoop

            app = Flask(__name__, static_url_path='/static')

            app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
            CORS(app)

            @app.route('/')
            def index():
                return app.send_static_file('index.html')

            @app.route('/files/<path:path>')
            def static_files(path):
                return app.send_static_file('files/' + path)

            @app.route('/api', methods=['GET'])
            def api():
                query = request.args['query']
                out = retrieve(query, 5)
                return jsonify(out)

            print('Starting server at %d' % args.port)
            http_server = HTTPServer(WSGIContainer(app))
            http_server.listen(args.port)
            IOLoop.instance().start()


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_arguments()
    args = argument_parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'embed' or args.mode == 'embed_context' or args.mode == 'embed_question':
        embed(args)
    elif args.mode.startswith('serve'):
        serve(args)
    else:
        raise Exception()


if __name__ == "__main__":
    from_ = importlib.import_module(sys.argv[1])
    ArgumentParser = from_.ArgumentParser
    FileInterface = from_.FileInterface
    Processor = from_.Processor
    Sampler = from_.Sampler
    Model = from_.Model
    Loss = from_.Loss
    assert issubclass(ArgumentParser, base.ArgumentParser)
    assert issubclass(FileInterface, base.FileInterface)
    assert issubclass(Processor, base.Processor)
    assert issubclass(Sampler, base.Sampler)
    assert issubclass(Model, base.Model)
    assert issubclass(Loss, base.Loss)
    main()
