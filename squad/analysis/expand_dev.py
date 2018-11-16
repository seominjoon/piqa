"""
Expand dev-1.1.json (or given) dataset using DrQA's Document Retriever

additional requirements:
    https://github.com/facebookresearch/DrQA
"""

import argparse
import os
import json
import numpy as np

from tqdm import tqdm


# Copied from ../baseline/file_interface
def _load_squad(squad_path, draft=False):
    with open(squad_path, 'r') as fp:
        squad = json.load(fp)
        examples = []
        for article in squad['data']:
            for para_idx, paragraph in enumerate(article['paragraphs']):
                cid = '%s_%d' % (article['title'], para_idx)
                if 'context' in paragraph:
                    context = paragraph['context']
                    context_example = {'cid': cid, 'context': context}
                else:
                    context_example = {}

                if 'qas' in paragraph:
                    for question_idx, qa in enumerate(paragraph['qas']):
                        id_ = qa['id']
                        qid = '%s_%d' % (cid, question_idx)
                        question = qa['question']
                        question_example = {'id': id_, 'qid': qid, 'question': question}
                        if 'answers' in qa:
                            answers, answer_starts, answer_ends = [], [], []
                            for answer in qa['answers']:
                                answer_start = answer['answer_start']
                                answer_end = answer_start + len(answer['text'])
                                answers.append(answer['text'])
                                answer_starts.append(answer_start)
                                answer_ends.append(answer_end)
                            answer_example = {'answers': answers, 'answer_starts': answer_starts,
                                              'answer_ends': answer_ends}
                            question_example.update(answer_example)

                        example = {'idx': len(examples)}
                        example.update(context_example)
                        example.update(question_example)
                        examples.append(example)
                        if draft and len(examples) == 100:
                            return examples
                else:
                    example = {'idx': len(examples)}
                    example.update(context_example)
                    examples.append(example)
                    if draft and len(examples) == 100:
                        return examples
        return examples


# Predefined paths
home = os.path.expanduser('~')
DATA_PATH = os.path.join(home, 'data/squad/dev-v1.1.json')
RET_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data', 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Expand script for analysis')
    parser.add_argument('--data-path', type=str, default=DATA_PATH,
                        help='Dataset file path')
    parser.add_argument('--retriever-path', type=str, default=RET_PATH,
                        help='Document Retriever path')
    parser.add_argument('--n-docs', type=int, default=10,
                        help='Number of closest documents')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process workers')
    args = parser.parse_args()

    # Load SQuAD data
    dev_data = _load_squad(args.data_path)
    print('Data from {} with size {}'.format(args.data_path, len(dev_data)))

    # Load DrQA retriever
    from drqa import retriever, tokenizers
    from drqa.retriever import utils

    # Test retriever
    ranker = retriever.get_class('tfidf')(tfidf_path=args.retriever_path)
    print('Retriever loaded from {}'.format(args.retriever_path))
    closest_docs = ranker.batch_closest_docs(
        ['Who won the heavy weight championship?'], 
        k=args.n_docs, num_workers=args.num_workers
    )
    print('Test result: {}\n'.format(closest_docs[0][0]))

    # Iterate SQuAD data and retrieve closest docs
    batch_size = 4
    for dev_idx in tqdm(range(0, len(dev_data), batch_size)):
        batch_context = [ex['context'] 
                         for ex in dev_data[dev_idx:dev_idx+batch_size]]
        closest_docs = ranker.batch_closest_docs(batch_context, 
                                                 k=args.n_docs)
        for i, dev_ex in enumerate(dev_data[dev_idx:dev_idx+batch_size]):
            dev_ex['closest_docs_{}'.format(args.n_docs)] = \
                (closest_docs[i][0], list(closest_docs[i][1]))

    # Check integrity and save
    print(dev_data[5])
    with open('dev-v1.1-top{}.json'.format(args.n_docs), 'w') as f:
        json.dump(json.dumps(dev_data), f)
    print('Saved!')

    # TODO: use doc_db to retrieve documents, and expand dev dataset

