"""Official merge script for PI-SQuAD v0.1"""
from __future__ import print_function

import os
import argparse
import json
import sys
import pickle

import scipy.sparse
import numpy as np

from scipy.sparse import csr_matrix, hstack, vstack, save_npz, load_npz
from tqdm import tqdm


# Append document tfidf vector to each paragraphs, then merge
def concat_merge_tfidf(c2q, context_emb_dir, doc_tfidf_dir,
                       question_emb_dir, que_tfidf_dir, mode, **kwargs):
    num_questions = sum([len(q) for q in c2q.values()])
    print('Number of contexts to process: {}'.format(len(c2q)))
    print('Number of questions to process: {}'.format(num_questions))

    # Merge using tfidf concatenation
    tfidf_weight = kwargs['tfidf_weight']

    # Load tfidf mats
    neg_doc_mat_path = os.path.join(doc_tfidf_dir, 'neg_doc_mat_tf.pkl')
    pos_doc_mat_path = os.path.join(doc_tfidf_dir, 'pos_doc_mat.pkl')
    assert os.path.exists(neg_doc_mat_path) and os.path.exists(pos_doc_mat_path)
    with open(neg_doc_mat_path, 'rb') as f:
        neg_doc_mat = pickle.load(f)
    with open(pos_doc_mat_path, 'rb') as f:
        pos_doc_mat = pickle.load(f)

    predictions = {}
    for cid, q_list in tqdm(c2q.items()):

        # Load metadata for pharse to doc matching
        metadata_path = os.path.join(context_emb_dir, cid + '.metadata')
        if not os.path.exists(metadata_path):
            continue
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'r') as fp:
            metadata = json.load(fp)

        # Load neg doc tfidf vectors [(N-1) X V]
        tfidf_vecs = []
        num_neg_doc = metadata['num_eval_par']
        for neg_idx in range(num_neg_doc):
            ndoc_title = metadata['context_src_{}'.format(neg_idx)]
            num_phrases = metadata['num_phrases_{}'.format(neg_idx)]
            if num_phrases == 0:
                print('No phrase:', metadata['context_{}'.format(neg_idx)], cid)

            assert ndoc_title in neg_doc_mat[0], ndoc_title
            neg_doc_idx = neg_doc_mat[0][ndoc_title]
            neg_vec = neg_doc_mat[2][neg_doc_idx]
            if mode == 'E':
                neg_vec = vstack(
                    [neg_vec] * metadata['num_phrases_{}'.format(neg_idx)]
                )
            tfidf_vecs.append(neg_vec)

        # Load pos doc tfidf vector [1 X V], and stack as [N X V]
        doc_title = '_'.join(cid.split('_')[:-1])
        assert doc_title in pos_doc_mat
        pos_vec = pos_doc_mat[doc_title]
        if mode == 'E':
            pos_vec = vstack(
                [pos_vec] * metadata['num_phrases_{}'.format(num_neg_doc)]
            )
        tfidf_vecs.append(pos_vec)
        tfidf_vec = vstack(tfidf_vecs)

        # Load phrase vectors (supports dense vectors only) [P X D]
        phrase_emb_path = os.path.join(context_emb_dir, cid + '.npz')
        assert os.path.exists(phrase_emb_path)
        if not kwargs['sparse']:
            phrase_emb = np.load(phrase_emb_path)['arr_0']
        else:
            phrase_emb = load_npz(phrase_emb_path).tocsr()
        if mode == 'E':
            assert phrase_emb.shape[0] == tfidf_vec.shape[0], '{} vs {}'.format(
                phrase_emb.shape, tfidf_vec.shape)

        # Load question tfidf vectors [N X V]
        que_tfidf_paths = [
            os.path.join(que_tfidf_dir, q_id + '.tfidf.npz')
            for q_id in q_list
        ]
        for path in que_tfidf_paths:
            assert os.path.exists(path)
        que_tfidf_vec = vstack(
            [load_npz(que_tfidf_path) for que_tfidf_path in que_tfidf_paths]
        ).tocsr()

        # Load question embedding vectors [N X D]
        que_emb_paths = [
            os.path.join(question_emb_dir, q_id + '.npz')
            for q_id in q_list
        ]
        for path in que_emb_paths:
            assert os.path.exists(path)
        if not kwargs['sparse']:
            que_emb = np.stack(
                [np.load(que_emb_path)['arr_0'] 
                 for que_emb_path in que_emb_paths],
            )
            que_emb = np.squeeze(que_emb, axis=1)
        else:
            que_emb = vstack(
                [load_npz(que_emb_path) for que_emb_path in que_emb_paths]
            ).tocsr()

        # Load json file to get raw texts
        c_json_path = os.path.join(context_emb_dir, cid + '.json')
        assert os.path.exists(c_json_path)
        with open(c_json_path, 'r') as fp:
            phrases = json.load(fp)
            assert len(phrases) == phrase_emb.shape[0]

        # End-2-end mode (concatenation)
        if mode == 'E':

            # Concatenate doc_tfidf_vec and phrase_emb
            tfidf_vec = tfidf_vec    
            phrase_emb = csr_matrix(phrase_emb)
            phrase_concat_emb = hstack([tfidf_vec, phrase_emb])

            # Concatenate que_tfidf_vec and que_emb
            que_tfidf_vec = que_tfidf_vec * tfidf_weight         
            que_emb = csr_matrix(que_emb)
            que_concat_emb = hstack([que_tfidf_vec, que_emb])

            # Find answers
            sim = que_concat_emb * phrase_concat_emb.T
            phrase_idxs = np.argmax(sim.toarray(), axis=1)
            assert len(phrase_idxs) == len(q_list)
            for (phrase_idx, q_id) in zip(phrase_idxs, q_list):
                predictions[q_id] = phrases[phrase_idx]

        # Pipeline mode (filtering)
        elif mode == 'P':
            
            # Find argmax of document first, and find phrase idx
            sim1 = np.argmax((que_tfidf_vec * tfidf_vec.T).toarray(), axis=1)
            phrase_idxs = [0]
            base_idx = 0
            for neg_idx in range(metadata['num_eval_par'] + 1):
                num_phrases = metadata['num_phrases_{}'.format(neg_idx)]
                base_idx += num_phrases
                phrase_idxs.append(base_idx)
            assert phrase_idxs[-1] == phrase_emb.shape[0]
            
            # Retrieval based on the phrase idxs
            for q_idx, q_id in enumerate(q_list):
                q_emb = que_emb[q_idx]
                masked_phrase_emb = phrase_emb[:]
                range_s = phrase_idxs[sim1[q_idx]]
                range_e = phrase_idxs[sim1[q_idx]+1]
                # No match (no phrase found for corresponding doc
                if range_s == range_e:
                    predictions[q_id] = ''
                    print('No match for qid {}'.format(q_id))
                    continue

                sim2 = np.argmax(
                    q_emb.dot(masked_phrase_emb[range_s:range_e,:].T)
                )
                sim2 += range_s
                assert sim2 >= range_s and sim2 < range_e
                predictions[q_id] = phrases[sim2]
        else:
            raise NotImplementedError()

        if kwargs['draft']:
            for q_id, pred in predictions.items():
                print(q_id, pred)
                exit(1)
         
    return predictions


if __name__ == '__main__':
    squad_expected_version = '1.1'
    parser = argparse.ArgumentParser(description='script for appending tf-idf')
    parser.add_argument('data_path', help='Dataset file path')
    parser.add_argument('context_emb_dir', help='Context embedding directory')
    parser.add_argument('doc_tfidf_dir', help='Document tfidf directory')
    parser.add_argument('question_emb_dir', help='Question embedding directory')
    parser.add_argument('que_tfidf_dir', help='Question tfidf directory')
    parser.add_argument('pred_path', help='Prediction json file path')
    parser.add_argument('--mode', type=str, default='E', help='E|P')
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='If stored phrase vecs are sparse vec or not')
    parser.add_argument('--tfidf-weight', type=float, default=1e+0,
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
