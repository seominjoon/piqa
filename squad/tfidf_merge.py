"""Official merge script for PI-SQuAD v0.1"""
from __future__ import print_function

import os
import argparse
import json
import sys

import scipy.sparse
import numpy as np

from scipy.sparse import csr_matrix, hstack, vstack, save_npz, load_npz
from tqdm import tqdm


# Append document tfidf vector to each paragraphs, then merge
def concat_merge_tfidf(c2q, context_emb_dir, question_emb_dir, **kwargs):
    num_questions = sum([len(q) for q in c2q.values()])
    print('Number of contexts to process: {}'.format(len(c2q)))
    print('Number of questions to process: {}'.format(num_questions))

    # Merge using tfidf concatenation
    tfidf_weight = kwargs['tfidf_weight']
    predictions = {}
    for cid, q_list in tqdm(c2q.items()):

        # Load doc tfidf vector [1 X V]
        doc_title = '_'.join(cid.split('_')[:-1])
        doc_tfidf_path = os.path.join(
            context_emb_dir,
            doc_title + '.tfidf.npz'
        )
        assert os.path.exists(doc_tfidf_path)
        doc_tfidf_emb = load_npz(doc_tfidf_path)

        # Load phrase vectors (supports dense vectors only) [P X D]
        phrase_emb_path = os.path.join(context_emb_dir, cid + '.npz')
        assert os.path.exists(phrase_emb_path)
        phrase_emb = np.load(phrase_emb_path)['arr_0']

        # Concatenate doc_tfidf_emb and phrase_emb
        # TODO: need to concatenate different doc_tfidf to each vector
        doc_tfidf_emb = doc_tfidf_emb * tfidf_weight         
        doc_tfidf_emb = vstack([doc_tfidf_emb] * phrase_emb.shape[0])
        phrase_emb = csr_matrix(phrase_emb)
        phrase_concat_emb = hstack([doc_tfidf_emb, phrase_emb])

        # Load question tfidf vectors [N X V]
        que_tfidf_paths = [
            os.path.join(question_emb_dir, q_id + '.tfidf.npz')
            for q_id in q_list
        ]
        for path in que_tfidf_paths:
            assert os.path.exists(path)
        que_tfidf_emb = vstack(
            [load_npz(que_tfidf_path) for que_tfidf_path in que_tfidf_paths]
        )

        # Load question embedding vectors [N X D]
        que_emb_paths = [
            os.path.join(question_emb_dir, q_id + '.npz')
            for q_id in q_list
        ]
        for path in que_emb_paths:
            assert os.path.exists(path)
        que_emb = np.stack(
            [np.load(que_emb_path)['arr_0'] for que_emb_path in que_emb_paths],
        )
        que_emb = np.squeeze(que_emb, axis=1)

        # Concatenate que_tfidf_emb and que_emb
        que_tfidf_emb = que_tfidf_emb * tfidf_weight         
        que_emb = csr_matrix(que_emb)
        que_concat_emb = hstack([que_tfidf_emb, que_emb])

        # Load json file to get raw texts
        c_json_path = os.path.join(context_emb_dir, cid + '.json')
        assert os.path.exists(c_json_path)
        with open(c_json_path, 'r') as fp:
            phrases = json.load(fp)
            assert len(phrases) == phrase_concat_emb.shape[0]

        # Find answers
        sim = que_concat_emb * phrase_concat_emb.T
        phrase_idxs = np.argmax(sim.toarray(), axis=1)
        assert len(phrase_idxs) == len(q_list)
        for (phrase_idx, q_id) in zip(phrase_idxs, q_list):
            predictions[q_id] = phrases[phrase_idx]

        if kwargs['draft']:
            break
         
    return predictions


if __name__ == '__main__':
    squad_expected_version = '1.1'
    parser = argparse.ArgumentParser(description='script for appending tf-idf')
    parser.add_argument('data_path', help='Dataset file path')
    parser.add_argument('context_emb_dir', help='Context embedding directory')
    parser.add_argument('question_emb_dir', help='Question embedding directory')
    parser.add_argument('pred_path', help='Prediction json file path')
    parser.add_argument('--tfidf-weight', type=float, default=1e+1,
                        help='TF-IDF vector weight')
    parser.add_argument('--draft', default=False, action='store_true',
                        help='Draft version')
    args = parser.parse_args()

    # Read squad dataset
    with open(args.data_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != squad_expected_version:
            print('Evaluation expects v-' + squad_expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    
    # Only supports c2q (q_mat) version
    from merge import get_c2q
    c2q = get_c2q(dataset)

    # Merge using tfidf
    predictions = concat_merge_tfidf(c2q, **args.__dict__)

    with open(args.pred_path, 'w') as fp:
        json.dump(predictions, fp)
