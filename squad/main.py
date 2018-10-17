import sys
import time
from collections import OrderedDict
from pprint import pprint
import importlib

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

    loss_model = Loss().to(device)
    optimizer = torch.optim.Adam(p for p in model.parameters() if p.requires_grad)

    interface.bind(processor, model, optimizer=optimizer)

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

                for id_, phrases, matrix in context_results:
                    interface.context_emb(id_, phrases, matrix, emb_type=args.emb_type)

            if args.mode == 'embed' or args.mode == 'embed_question':

                question_output = model.get_question(**test_batch)
                question_results = processor.postprocess_question_batch(test_dataset, test_batch, question_output)

                for id_, emb in question_results:
                    interface.question_emb(id_, emb, emb_type=args.emb_type)

            print('[%d/%d]' % (batch_idx + 1, len(test_loader)))


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
