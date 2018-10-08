import argparse
import time
import os
from collections import OrderedDict
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import DataLoader

from baseline.model import Model, Loss
from baseline.processor import Processor, Sampler
from baseline.file_interface import FileInterface


def get_args():
    home = os.path.expanduser('~')
    parser = argparse.ArgumentParser(description='PIQA')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)  # ignore this argument.

    # Data (input) paths
    parser.add_argument('--train_path', type=str, default=os.path.join(home, 'data', 'squad', 'train-v1.1.json'),
                        help='location of the training data')
    parser.add_argument('--test_path', type=str, default=os.path.join(home, 'data', 'squad', 'dev-v1.1.json'),
                        help='location of the test data')
    parser.add_argument('--glove_dir', type=str, default=os.path.join(home, 'data', 'glove'),
                        help='location of GloVe')
    parser.add_argument('--elmo_options_file', type=str, default=os.path.join(home, 'data', 'elmo', 'options.json'))
    parser.add_argument('--elmo_weights_file', type=str, default=os.path.join(home, 'data', 'elmo', 'weights.hdf5'))

    # Output paths
    parser.add_argument('--output_dir', type=str, default='/tmp/piqa', help='Output directory')
    parser.add_argument('--save_dir', type=str, default=None, help='location for saving the model')
    parser.add_argument('--load_dir', type=str, default=None, help='location for loading the model')
    parser.add_argument('--dump_dir', type=str, default=None, help='location for dumping outputs')
    parser.add_argument('--report_path', type=str, default=None, help='location for report')
    parser.add_argument('--pred_path', type=str, default=None, help='location for prediction json file during `test`')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--question_emb_dir', type=str, default=None)
    parser.add_argument('--context_emb_dir', type=str, default=None)

    # Model arguments
    parser.add_argument('--word_vocab_size', type=int, default=10000)
    parser.add_argument('--char_vocab_size', type=int, default=100)
    parser.add_argument('--glove_vocab_size', type=int, default=400002)
    parser.add_argument('--glove_size', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--elmo', default=False, action='store_true')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--max_pool', default=False, action='store_true')
    parser.add_argument('--agg', type=str, default='max', help='max|logsumexp')
    parser.add_argument('--num_layers', type=int, default=1)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_steps', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--eval_save_period', type=int, default=500)
    parser.add_argument('--report_period', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Other arguments
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--glove_cuda', default=False, action='store_true')
    parser.add_argument('--cache', default=False, action='store_true')
    parser.add_argument('--preload', default=False, action='store_true')
    parser.add_argument('--dump_period', type=int, default=20)
    parser.add_argument('--emb_type', type=str, default='dense')

    args = parser.parse_args()

    # modify args
    if args.draft:
        args.glove_vocab_size = 102
        args.batch_size = 2
        args.eval_steps = 1
        args.eval_save_period = 2
        args.train_steps = 2

    args.embed_size = args.glove_size
    args.glove_cpu = not args.glove_cuda

    if args.save_dir is None:
        args.save_dir = os.path.join(args.output_dir, 'save')
    if args.load_dir is None:
        args.load_dir = os.path.join(args.output_dir, 'save')
    if args.dump_dir is None:
        args.dump_dir = os.path.join(args.output_dir, 'dump')
    if args.question_emb_dir is None:
        args.question_emb_dir = os.path.join(args.output_dir, 'question_emb')
    if args.context_emb_dir is None:
        args.context_emb_dir = os.path.join(args.output_dir, 'context_emb')
    if args.report_path is None:
        args.report_path = os.path.join(args.output_dir, 'report.csv')
    if args.pred_path is None:
        args.pred_path = os.path.join(args.output_dir, 'pred.json')
    if args.cache_path is None:
        args.cache_path = os.path.join(args.output_dir, 'cache.b')

    return args


def get_dump(dataset, input_, output, results):
    dump = []
    for i, idx in enumerate(input_['idx']):
        example = dataset[idx]
        each = {'id': example['id'],
                'context': example['context'],
                'question': example['question'],
                'answer_starts': example['answer_starts'],
                'answer_ends': example['answer_ends'],
                'context_spans': example['context_spans'],
                'yp1': output['yp1'][i].cpu().numpy(),
                'yp2': output['yp2'][i].cpu().numpy(),
                }
        dump.append(each)
    return dump


def preprocess(interface, args):
    """Helper function for caching preprocessed data
    """
    # get data
    print('Loading train and dev data')
    train_examples = interface.load_train()
    dev_examples = interface.load_test()

    # iff creating processor
    print('Loading metadata')
    metadata = interface.load_metadata()

    print('Constructing processor')
    processor = Processor(**args.__dict__)
    processor.construct(train_examples, metadata=metadata)

    # data loader
    print('Preprocessing datasets and metadata')
    train_dataset = tuple(processor.preprocess(example) for example in train_examples)
    dev_dataset = tuple(processor.preprocess(example) for example in dev_examples)
    processed_metadata = processor.process_metadata(metadata)

    print('Creating data loaders')
    train_sampler = Sampler(train_dataset, max_context_size=256, max_question_size=32, bucket=True,
                            shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=processor.collate, sampler=train_sampler)

    dev_sampler = Sampler(dev_dataset, bucket=True)
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
    model = Model(**args.__dict__).to(device)

    loss_model = Loss().to(device)
    optimizer = torch.optim.Adam(p for p in model.parameters() if p.requires_grad)

    out = interface.cache(preprocess, args) if args.cache else preprocess(interface, args)
    processor = out['processor']
    processed_metadata = out['processed_metadata']
    train_dataset = out['train_dataset']
    dev_dataset = out['dev_dataset']
    train_loader = out['train_loader']
    dev_loader = out['dev_loader']
    interface.bind(processor, model, optimizer=optimizer)

    model.init(processed_metadata)

    step = 0
    train_report, dev_report = None, None

    print('Training')
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
    model = Model(**args.__dict__).to(device)
    processor = Processor(**args.__dict__)
    interface.bind(processor, model)

    interface.load(args.iteration, session=args.load_dir)

    test_examples = interface.load_test()
    test_dataset = tuple(processor.preprocess(example) for example in test_examples)

    test_sampler = Sampler(test_dataset, bucket=True)
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
                dump = get_dump(test_dataset, test_batch, model_output, results)
                interface.dump(batch_idx, dump)
            for result in results:
                pred[result['id']] = result['pred']

            print('[%d/%d]' % (batch_idx + 1, len(test_loader)))
        interface.pred(pred)


def embed(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    pprint(args.__dict__)

    interface = FileInterface(**args.__dict__)
    model = Model(**args.__dict__).to(device)
    processor = Processor(**args.__dict__)
    interface.bind(processor, model)

    interface.load(args.iteration, session=args.load_dir)

    test_examples = interface.load_test()
    test_dataset = tuple(processor.preprocess(example) for example in test_examples)

    test_sampler = Sampler(test_dataset, bucket=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             collate_fn=processor.collate)

    print('Saving embeddings')
    with torch.no_grad():
        model.eval()
        for batch_idx, (test_batch, _) in enumerate(zip(test_loader, range(args.eval_steps))):
            test_batch = {key: val.to(device) for key, val in test_batch.items()}

            if args.mode == 'embed' or args.mode == 'embed_context':

                context_output = model.get_context(**test_batch)
                context_results = processor.postprocess_context_batch(test_dataset, test_batch, context_output,
                                                                      emb_type=args.emb_type)

                for id_, phrases, matrix in context_results:
                    interface.context_emb(id_, phrases, matrix, emb_type=args.emb_type)

            if args.mode == 'embed' or args.mode == 'embed_question':

                question_output = model.get_question(**test_batch)
                question_results = processor.postprocess_question_batch(test_dataset, test_batch, question_output,
                                                                        emb_type=args.emb_type)

                for id_, emb in question_results:
                    interface.question_emb(id_, emb, emb_type=args.emb_type)

            print('[%d/%d]' % (batch_idx + 1, len(test_loader)))


def main():
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'embed' or args.mode == 'embed_context' or args.mode == 'embed_question':
        embed(args)
    else:
        raise Exception()


if __name__ == "__main__":
    main()
