import argparse
import json
import time
import os
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader

from model import PIQA, Loss
from data import load_glove, load_squad, SquadProcessor, SquadSampler
from file import FileInterface


def get_args():
    home = os.path.expanduser('~')
    parser = argparse.ArgumentParser(description='PIQA')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)  # ignore this argument.

    # Paths
    parser.add_argument('--data_dir', type=str, default=os.path.join(home, 'data/squad/'),
                        help='location of the data corpus')
    parser.add_argument('--save_dir', type=str, default='/tmp/piqa', help='location for saving the model')
    parser.add_argument('--load_dir', type=str, default='/tmp/piqa', help='location for loading the model')
    parser.add_argument('--glove_dir', type=str, default=os.path.join(home, 'data', 'glove'),
                        help='location of GloVe')
    parser.add_argument('--pred_path', type=str, default='/tmp/piqa/pred.json')
    parser.add_argument('--question_emb_path', type=str, default='/tmp/piqa/ques_emb.hdf5')
    parser.add_argument('--context_emb_path', type=str, default='/tmp/piqa/context_emb.hdf5')
    parser.add_argument('--context_phrase_path', type=str, default='/tmp/piqa/context_phrase.json')
    parser.add_argument('--elmo_options_file', type=str, default=None)
    parser.add_argument('--elmo_weight_file', type=str, default=None)

    # Model-specific arguments
    parser.add_argument('--word_vocab_size', type=int, default=100000)
    parser.add_argument('--char_vocab_size', type=int, default=100)
    parser.add_argument('--glove_vocab_size', type=int, default=400002)
    parser.add_argument('--glove_size', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--output_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--elmo', default=False, action='store_true')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_save_period', type=int, default=500)
    parser.add_argument('--report_period', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Other arguments
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--load_cache', default=False, action='store_true')
    parser.add_argument('--save_cache', default=False, action='store_true')

    args = parser.parse_args()

    # modify args
    if args.draft:
        args.glove_vocab_size = 102

    return args


def bind_model(interface, processor, model, optimizer=None):
    def load(filename, **kwargs):
        # filename = os.path.join(filename, 'model.pt')
        state = torch.load(filename)
        processor.load_state_dict(state['preprocessor'])
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded from %s' % filename)

    def save(filename, **kwargs):
        state = {
            'preprocessor': processor.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        filename = os.path.join(filename, 'model.pt')
        torch.save(state, filename)
        print('Model saved at %s' % filename)

    def infer(input, top_k=100):
        # input = {'id': '', 'question': '', 'context': ''}
        model.eval()

    interface.bind(save=save, load=load)


def train(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    interface = FileInterface(args.data_dir, args.glove_dir, args.save_dir, args.pred_path, args.question_emb_path)

    piqa_model = PIQA(args.char_vocab_size,
                      args.glove_vocab_size,
                      args.word_vocab_size,
                      args.hidden_size,
                      args.glove_size,
                      args.dropout,
                      args.num_heads,
                      elmo=args.elmo,
                      elmo_options_file=args.elmo_options_file,
                      elmo_weight_file=args.elmo_weight_file).to(device)
    piqa_model.embedding.glove_embedding = piqa_model.embedding.glove_embedding.cpu()
    loss_model = Loss()
    optimizer = torch.optim.Adam(p for p in piqa_model.parameters() if p.requires_grad)

    if not args.load_cache or not os.path.exists('cache'):
        # get data
        print('Loading train and dev data')
        train_examples = load_squad(os.path.join(interface.data_dir, 'train-v1.1.json'), draft=args.draft)
        dev_examples = load_squad(os.path.join(interface.data_dir, 'dev-v1.1.json'), draft=args.draft)

        # iff creating processor
        print('Loading GloVe')
        glove_words, glove_emb_mat = load_glove(args.glove_size, glove_dir=interface.glove_dir, draft=args.draft)

        print('Constructing processor')
        processor = SquadProcessor(args.char_vocab_size, args.glove_vocab_size, args.word_vocab_size, elmo=args.elmo)
        processor.construct(train_examples, glove_words)

        # data loader
        print('Preprocessing datasets')
        train_dataset = tuple(processor.preprocess(example) for example in train_examples)
        dev_dataset = tuple(processor.preprocess(example) for example in dev_examples)

        print('Creating data loaders')
        train_sampler = SquadSampler(train_dataset, max_context_size=256, max_question_size=32, bucket=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  collate_fn=processor.collate, sampler=train_sampler)
        train_loader = tuple(train_loader)

        dev_sampler = SquadSampler(dev_dataset, bucket=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                collate_fn=processor.collate, sampler=dev_sampler)
        dev_loader = tuple(dev_loader)

        if args.save_cache:
            print('Caching')
            cache = {'glove_emb_mat': glove_emb_mat,
                     'processor': processor,
                     'train_dataset': train_dataset,
                     'dev_dataset': dev_dataset,
                     'train_loader': train_loader,
                     'dev_loader': dev_loader}
            torch.save(cache, 'cache')
    else:
        print('Loading cache')
        cache = torch.load('cache')
        glove_emb_mat = cache['glove_emb_mat']
        processor = cache['processor']
        train_dataset = cache['train_dataset']
        dev_dataset = cache['dev_dataset']
        train_loader = cache['train_loader']
        dev_loader = cache['dev_loader']

    print("Initializing model weights")
    piqa_model.load_glove(torch.tensor(glove_emb_mat))

    bind_model(interface, processor, piqa_model, optimizer=optimizer)

    step = 0
    start_time = time.time()
    best_report = None

    print('Training')
    piqa_model.train()
    for epoch_idx in range(args.epochs):
        for i, train_batch in enumerate(train_loader):
            train_batch = {key: val.to(device) for key, val in train_batch.items()}
            model_output = piqa_model(**train_batch)
            train_results = processor.postprocess_batch(train_dataset, train_batch, model_output)
            train_loss = loss_model(model_output[0], model_output[1], **train_batch)
            train_f1 = float(np.mean([result['f1'] for result in train_results]))
            train_em = float(np.mean([result['em'] for result in train_results]))

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1

            # report & eval & save
            if step % args.report_period == 1:
                report = OrderedDict(step=step, train_loss=train_loss.item(), train_f1=train_f1, train_em=train_em,
                                     time=time.time() - start_time)
                interface.report(**report)
                print(', '.join('%s=%.5r' % (s, r) for s, r in report.items()))

            if step % args.eval_save_period == 1:
                with torch.no_grad():
                    piqa_model.eval()
                    pred = {}
                    dev_losses, dev_results = [], []
                    for dev_batch, _ in zip(dev_loader, range(args.eval_steps)):
                        dev_batch = {key: val.to(device) for key, val in dev_batch.items()}
                        model_output = piqa_model(**dev_batch)
                        results = processor.postprocess_batch(dev_dataset, dev_batch, model_output)

                        dev_loss = loss_model(model_output[0], model_output[1], **dev_batch)

                        for result in results:
                            pred[result['id']] = result['pred']
                        dev_results.extend(results)
                        dev_losses.append(dev_loss.item())

                    dev_loss = float(np.mean(dev_losses))
                    dev_f1 = float(np.mean([result['f1'] for result in dev_results]))
                    dev_em = float(np.mean([result['em'] for result in dev_results]))

                    report = OrderedDict(step=step, dev_loss=dev_loss, dev_f1=dev_f1, dev_em=dev_em,
                                         time=time.time() - start_time)
                    summary = False
                    if best_report is None or report['dev_f1'] > best_report['dev_f1']:
                        best_report = report
                        summary = True
                        interface.save(iteration=step)
                        interface.pred(pred)
                    interface.report(summary=summary, **report)
                    print(', '.join('%s=%.5r' % (s, r) for s, r in report.items()),
                          '(dev_f1_best=%.5r @%d)' % (best_report['dev_f1'], best_report['step']))
                    piqa_model.train()


def test(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    interface = FileInterface(args.data_dir, args.glove_dir, args.save_dir, args.pred_path, args.question_emb_path)

    piqa_model = PIQA(args.char_vocab_size,
                      args.glove_vocab_size,
                      args.word_vocab_size,
                      args.hidden_size,
                      args.glove_size,
                      args.dropout,
                      args.num_heads).to(device)
    piqa_model.embedding.glove_embedding = piqa_model.embedding.glove_embedding.cpu()

    processor = SquadProcessor(args.char_vocab_size, args.glove_vocab_size, args.word_vocab_size)

    bind_model(interface, processor, piqa_model)
    interface.load(args.iteration, session=args.load_dir)

    test_examples = load_squad(os.path.join(interface.data_dir, 'test'), draft=args.draft)
    test_dataset = tuple(processor.preprocess(example) for example in test_examples)

    test_sampler = SquadSampler(test_dataset, bucket=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             collate_fn=processor.collate)

    with torch.no_grad():
        piqa_model.eval()
        pred = {}
        for test_batch, _ in zip(test_loader, range(args.eval_steps)):
            test_batch = {key: val.to(device) for key, val in test_batch.items()}
            model_output = piqa_model(**test_batch)
            results = processor.postprocess_batch(test_dataset, test_batch, model_output)
            for result in results:
                pred[result['id']] = result['pred']
        interface.pred(pred)


def test2(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    interface = FileInterface(args.data_dir, args.glove_dir, args.save_dir, args.pred_path, args.question_emb_path)

    piqa_model = PIQA(args.char_vocab_size,
                      args.glove_vocab_size,
                      args.word_vocab_size,
                      args.hidden_size,
                      args.glove_size,
                      args.dropout,
                      args.num_heads).to(device)
    piqa_model.embedding.glove_embedding = piqa_model.embedding.glove_embedding.cpu()

    processor = SquadProcessor(args.char_vocab_size, args.glove_vocab_size, args.word_vocab_size)

    bind_model(interface, processor, piqa_model)
    interface.load(args.iteration, session=args.load_dir)

    test_examples = load_squad(os.path.join(interface.data_dir, 'test'), draft=args.draft)
    test_dataset = tuple(processor.preprocess(example) for example in test_examples)

    test_sampler = SquadSampler(test_dataset, bucket=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             collate_fn=processor.collate)

    with torch.no_grad():
        piqa_model.eval()
        for test_batch, _ in zip(test_loader, range(args.eval_steps)):
            test_batch = {key: val.to(device) for key, val in test_batch.items()}
            context_output = piqa_model.context(**test_batch)
            context_results = processor.postprocess_context_batch(test_dataset, test_batch, context_output)

            question_output = piqa_model.question(**test_batch)
            question_results = processor.postprocess_question_batch(test_dataset, test_batch, question_output)

            for id_, emb in question_results:
                interface.question_emb(id_, emb)


if __name__ == "__main__":
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'test2':
        test2(args)
    else:
        raise Exception()
