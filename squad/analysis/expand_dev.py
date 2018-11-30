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
import random

from tqdm import tqdm


# Copied and modified from ../baseline/file_interface
def _load_squad_without_questions(squad_path, draft=False):
    with open(squad_path, 'r') as fp:
        squad = json.load(fp)

        # Remove questions (copied from ../split.py)
        for article in squad['data']:
            for para in article['paragraphs']:
                del para['qas']

        examples = []
        for article in squad['data']:
            for para_idx, paragraph in enumerate(article['paragraphs']):
                cid = '%s_%d' % (article['title'], para_idx)
                if 'context' in paragraph:
                    context = paragraph['context']
                    context_example = {'cid': cid, 'context': context}
                else:
                    context_example = {}

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


# Load unique squad docs and questions (separately)
def squad_docs_ques(squad, title_split='_', include_q=True):
    squad_docs = {}
    squad_ques = {}
    for item in squad:
        title = title_split.join(item['cid'].split('_')[:-1])
        if title not in squad_docs:
            squad_docs[title] = list()
        if item['context'] not in squad_docs[title]:
            squad_docs[title].append(item['context'])
        if include_q:
            squad_ques[item['id']] = item['question']
    return squad_docs, squad_ques


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
    parser.add_argument('--n-pars', type=int, default=100,
                        help='Number of closest paragraphs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process workers')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed (for reproducibility)')
    parser.add_argument('--find-docs', default=False, action='store_true',
                        help='True to find closest docs') 
    parser.add_argument('--mode', type=str, default='large', 
                        help='large|open')
    parser.add_argument('--par-open', default=False, action='store_true',
                        help='True to set n-docs-max as n-pars')
    parser.add_argument('--tfidf-sort', default=False, action='store_true',
                        help='Sort n-pars based on tfidf (random if false)')
    parser.add_argument('--draft', default=False, action='store_true')
    args = parser.parse_args()

    # Print important arguments
    if args.find_docs:
        print('Finding closest docs using DocumentRetriever')
        print('# of closest docs: {}'.format(args.n_docs))
    else:
        print('Expanding SQuAD development set')
        print('# of used closest docs: {}'.format(args.n_docs_max))
        print('# of noise paragraphs: {}'.format(args.n_pars))
        print('mode: {}'.format(args.mode))
        print('tfidf-sort: {}'.format(args.tfidf_sort))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load DrQA retriever
    from drqa import retriever, tokenizers
    from drqa.retriever import utils

    ##### Find top n closest docs #####
    if args.find_docs:

        # Load SQuAD data
        dev_data = _load_squad_without_questions(args.data_path)
        print('Data from {} with size {}'.format(args.data_path, len(dev_data)))

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
        with open('dev-v1.1-top{}.json'.format(args.n_docs), 'w') as f:
            json.dump(dev_data, f)
        print('Getting top {} similar docs done'.format(args.n_docs))
        exit(0)

    # Load Wikipedia DB
    db = retriever.DocDB(db_path=args.db_path)

    # Read result file, which must contain 'closest_docs_n' key
    with open('results/dev-v1.1-top{}.json'.format(args.n_docs), 'r') as f:
        squad = json.load(f)
        assert len(squad) > 0
        assert 'closest_docs_{}'.format(args.n_docs) in squad[0].keys()

    # Gather squad documents
    squad_docs, _ = squad_docs_ques(squad, title_split=' ', include_q=False)

    # For draft version
    if args.draft:
        squad = squad[:5]

    ##### For SQuAD-Open and SQuAD-Large-tfidf, we need closest docs #####
    open_context = set() 
    if args.mode == 'open' or (args.mode == 'large' and args.tfidf_sort):

        # Iterate dev-squad, and append retrieved docs
        for item in tqdm(squad):
            eval_context = [] # context for evaluation
            eval_context_src = []

            # Append SQuAD document for the title
            title = ' '.join(item['cid'].split('_')[:-1])
            assert title in squad_docs
            for par in squad_docs[title]:
                open_context.update([par])
            assert item['context'] in open_context

            # Add closest paragraphs with title filtering 
            doc_counter = 0
            for split_title in item['closest_docs_{}'.format(
                                    args.n_docs)][0]:
                text = db.get_doc_text(split_title)

                # Skip short text
                if all([len(par) < 500 for par in _split_doc(text)]):
                    continue

                for split_idx, split in enumerate(_split_doc(text)):
                    if split_idx == 0: # skip titles
                        curr_title = split
                        continue

                    # For SQuAD doc, we already added it.
                    if split_title == title or curr_title == title:
                        doc_counter -= 1
                        break

                    # Or, use what we've retrieved
                    else:
                        if len(split) >= 500:
                            eval_context.append(split)
                            eval_context_src.append(split_title)
                            if not args.par_open:
                                open_context.update([split])

                doc_counter += 1
                if doc_counter == args.n_docs_max:
                    break
            item['eval_context'] = eval_context
            item['eval_context_src'] = eval_context_src
            assert len(eval_context) == len(eval_context_src)
        print('Retrieving closest doc texts done')

    # Load retriever
    ranker = retriever.get_class('tfidf')(tfidf_path=args.retriever_path,
                                          strict=False)
    print('Retriever loaded from {}'.format(args.retriever_path))

    # Sort and filter paragraphs using TF-IDF
    if args.tfidf_sort:
        from scipy.sparse import vstack

        for item in tqdm(squad):
            # Filter the same context
            if item['context'] in item['eval_context']:
                same_idx = item['eval_context'].index(item['context'])
                del item['eval_context'][same_idx]
                del item['eval_context_src'][same_idx]

            # Calculate sparse vectors, and TF-IDF scores 
            eval_spvec = [ranker.text2spvec(t) 
                          for t in item['eval_context']]
            context_spvec = ranker.text2spvec(item['context'])
            scores = context_spvec * vstack(eval_spvec).T

            # Sorting using argsort
            if len(scores.data) <= args.n_pars:
                o_sort = np.argsort(-scores.data)
            else:
                o = np.argpartition(-scores.data, 
                                    args.n_pars)[0:args.n_pars]
                o_sort = o[np.argsort(-scores.data[o])]

            par_scores = scores.data[o_sort] # Not used
            par_ids = [i for i in scores.indices[o_sort]]

            # Update contexts
            selected_pars = list(np.array(item['eval_context'])[par_ids])
            selected_srcs = list(np.array(item['eval_context_src'])[par_ids])
            if len(selected_pars) < args.n_pars:
                print('Warning: not enough # par {}'.format(len(selected_pars)))
            if args.par_open and args.mode == 'open':
                open_context.update(selected_pars[:])
            elif args.mode == 'large':
                item['eval_context'] = selected_pars[:]
                item['eval_context_src'] = selected_srcs[:]

    # Or, we just use random docs
    else:
        for item in tqdm(squad):
            random_docs = np.random.randint(len(ranker.doc_dict[0]), 
                                            size=args.n_pars)
            new_docs = []
            for doc in random_docs:
                doc_title = ranker.get_doc_id(doc)
                par_lens = [
                    len(par) for par in _split_doc(db.get_doc_text(doc_title))
                ]

                # Avoid same docs, short docs
                while doc_title in squad_docs or \
                    all([par_len < 500 for par_len in par_lens]):
                    doc = (doc + 1) % len(ranker.doc_dict[0])
                    doc_title = ranker.get_doc_id(doc)
                    par_lens = [len(par) 
                        for par in _split_doc(db.get_doc_text(doc_title))]

                assert doc_title not in squad
                assert any([par_len >= 500 for par_len in par_lens])
                new_docs.append(doc)
            
            # Get random texts
            random_titles = [ranker.get_doc_id(doc) for doc in new_docs]
            random_texts = [list(_split_doc(db.get_doc_text(title)))[1:]
                            for title in random_titles]
            random_texts = [[par for par in text if len(par) >= 500]
                            for text in random_texts]
            assert all([all([len(par) >= 500 for par in text]) 
                        for text in random_texts])

            # Update contexts
            selected_pars = [
                np.random.choice(text, 1)[0] for text in random_texts
            ]
            assert len(selected_pars) == args.n_pars
            if args.mode == 'open':
                open_context.update(selected_pars[:])
            elif args.mode == 'large':
                item['eval_context'] = selected_pars[:]
                item['eval_context_src'] = random_titles[:]

    # For SQuAD-Large-(tfidf|rand)
    if args.mode == 'large':
        save_str = 'dev-v1.1-large-{}-{}par{}{}.json'.format(
            'tfidf' if args.tfidf_sort else 'rand',
            'doc{}-'.format(args.n_docs_max) if args.tfidf_sort else '',
            args.n_pars,
            '-draft' if args.draft else ''
        )
        with open(save_str, 'w') as f:
            json.dump(squad, f)

    # For SQuAD-Open
    elif args.mode == 'open':
        save_str = 'dev-v1.1-open-doc{}{}{}.json'.format(
            args.n_docs_max,
             '-par{}'.format(args.n_pars) if args.par_open else '',
             '-draft' if args.draft else ''
        )
        with open(save_str, 'w') as f:
            json.dump(list(open_context), f)

    print('SQuAD development set augmentation saved as {}'.format(save_str))

