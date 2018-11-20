"""
Expand dev-1.1.json (or given) dataset using DrQA's Document Retriever

additional requirements:
    https://github.com/facebookresearch/DrQA
"""

import argparse
import regex
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
                        question_example = {'id': id_, 'qid': qid, 
                                            'question': question}
                        if 'answers' in qa:
                            answers, answer_starts, answer_ends = [], [], []
                            for answer in qa['answers']:
                                answer_start = answer['answer_start']
                                answer_end = answer_start + len(answer['text'])
                                answers.append(answer['text'])
                                answer_starts.append(answer_start)
                                answer_ends.append(answer_end)
                            answer_example = {'answers': answers, 
                                              'answer_starts': answer_starts,
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


# Copeid from DrQA/drqa/pipeline/drqa.py
def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > 0:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)


# Predefined paths
home = os.path.expanduser('~')
DATA_PATH = os.path.join(home, 'data/squad/dev-v1.1.json')
RET_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data', 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
DB_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data',
    'wikipedia/docs.db')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Expand script for analysis')
    parser.add_argument('--data-path', type=str, default=DATA_PATH,
                        help='Dataset file path')
    parser.add_argument('--retriever-path', type=str, default=RET_PATH,
                        help='Document Retriever path')
    parser.add_argument('--db-path', type=str, default=DB_PATH,
                        help='Wikipedia DB path')
    parser.add_argument('--n-docs', type=int, default=100,
                        help='Number of total closest documents per ex')
    parser.add_argument('--n-docs-max', type=int, default=10,
                        help='Number of maximum closest documents per ex')
    parser.add_argument('--n-pars', type=int, default=10,
                        help='Number of closest paragraphs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process workers')
    parser.add_argument('--find-docs', default=False, action='store_true') 
    args = parser.parse_args()

    # Load SQuAD data
    dev_data = _load_squad(args.data_path)
    print('Data from {} with size {}'.format(args.data_path, len(dev_data)))

    # Load DrQA retriever
    from drqa import retriever, tokenizers
    from drqa.retriever import utils

    # Find top n docs?
    if args.find_docs:

        # Test retriever
        ranker = retriever.get_class('tfidf')(tfidf_path=args.retriever_path,
                                              strict=False)
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
        # print(dev_data[5])
        with open('dev-v1.1-top{}.json'.format(args.n_docs), 'w') as f:
            json.dump(dev_data, f)
        print('Getting top {} similar docs done.'.format(args.n_docs))

    # Use doc_db to retrieve documents, and expand dev dataset
    else:

        # Load Wikipedia DB
        db = retriever.DocDB(db_path=args.db_path)

        # Read result file, which must contain 'closest_docs_n' key
        with open('results/dev-v1.1-top{}.json'.format(args.n_docs), 'r') as f:
            squad = json.load(f)
            assert len(squad) > 0
            assert 'closest_docs_{}'.format(args.n_docs) in squad[0].keys()

            # Gather squad documents
            squad_docs = {}
            for item in squad:
                title = ' '.join(item['cid'].split('_')[:-1])
                if title not in squad_docs:
                    squad_docs[title] = list()
                squad_docs[title].append(item['context'])

            # Make it unique (preserving orders)
            # https://stackoverflow.com/a/480227
            seen = set()
            seen_add = seen.add
            unique_squad_docs = {key: [x for x in val 
                                       if not (x in seen or seen_add(x))]
                                 for key, val in squad_docs.items()}

            # Iterate dev-squad, and append retrieved docs
            open_context = set() # context for open-domain setting
            for item in tqdm(squad):
                eval_context = [] # context for evaluation

                # Append SQuAD document for the title
                title = ' '.join(item['cid'].split('_')[:-1])
                assert title in unique_squad_docs
                for par in unique_squad_docs[title]:
                    eval_context.append(par)
                    open_context.update([par])
                assert item['context'] in open_context
                assert item['context'] in eval_context

                # Iterate each closest doc with its text
                doc_counter = 0
                for split_title in item['closest_docs_{}'.format(
                                        args.n_docs)][0]:
                    text = db.get_doc_text(split_title)
                    for split_idx, split in enumerate(_split_doc(text)):
                        if split_idx == 0: # skip titles
                            curr_title = split
                            # TODO debugging (split title != curr_title)
                            continue

                        # For SQuAD, we already added it.
                        if split_title == title or curr_title == title:
                            doc_counter -= 1
                            break

                        # Or, use what we've retrieved
                        else:
                            eval_context.append(split)
                            open_context.update([split])

                    doc_counter += 1
                    if doc_counter == args.n_docs_max:
                        break

                item['eval_context'] = eval_context

        # Load retriever
        ranker = retriever.get_class('tfidf')(tfidf_path=args.retriever_path,
                                              strict=False)
        print('Retriever loaded from {}'.format(args.retriever_path))

        # Sort each paragraph using TF-IDF
        from scipy.sparse import vstack

        cache = {}
        for item in tqdm(squad):
            # Use cache for duplicate contexts
            if item['context'] not in cache:

                # Calculate sparse vectors, and TF-IDF scores 
                eval_spvec = [ranker.text2spvec(t) 
                              for t in item['eval_context']
                              if t != item['context']] 
                context_spvec = ranker.text2spvec(item['context'])
                scores = context_spvec * vstack(eval_spvec).T

                # Sorting copied from DrQA/drqa/retriever/tfidf_doc_ranker.py
                if len(scores.data) <= args.n_pars:
                    o_sort = np.argsort(-socres.data)
                else:
                    o = np.argpartition(-scores.data, 
                                        args.n_pars)[0:args.n_pars]
                    o_sort = o[np.argsort(-scores.data[o])]

                par_scores = scores.data[o_sort] # Not used
                par_ids = [i for i in scores.indices[o_sort]]

                # Assign sorted paragraphs
                item['eval_context'] = list(np.array(
                                            item['eval_context'])[par_ids])
                cache[item['context']] = item['eval_context'][:]
            else:
                item['eval_context'] = cache[item['context']][:]

        # Check integrity and save
        # print(squad[5])
        with open('dev-v1.1-top{}-eval-par{}.json'.format(args.n_docs,
                  args.n_pars), 'w') as f:
            json.dump(squad, f)
        with open('dev-v1.1-top{}-open-doc{}.json'.format(args.n_docs,
                  args.n_docs_max), 'w') as f:
            json.dump(list(open_context), f)
        print('SQuAD development set augmentation done.')

