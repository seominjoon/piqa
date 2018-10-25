""" Official alpha evaluation script for PIQA (inherited from SQuAD v1.1 evaluation script)."""
from __future__ import print_function

import os
from collections import Counter
import string
import re
import argparse
import json
import sys
import shutil

import scipy.sparse
import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def get_q2c(dataset):
    q2c = {}
    for article in dataset:
        for para_idx, paragraph in enumerate(article['paragraphs']):
            cid = '%s_%d' % (article['title'], para_idx)
            for qa in paragraph['qas']:
                q2c[qa['id']] = cid
    return q2c


def get_predictions(context_emb_path, question_emb_path, q2c, sparse=False, progress=False):
    context_emb_dir, context_emb_ext = os.path.splitext(context_emb_path)
    question_emb_dir, question_emb_ext = os.path.splitext(question_emb_path)
    if context_emb_ext == '.zip':
        print('Extracting %s to %s' % (context_emb_path, context_emb_dir))
        shutil.unpack_archive(context_emb_path, context_emb_dir)
    if question_emb_ext == '.zip':
        print('Extracting %s to %s' % (question_emb_path, question_emb_dir))
        shutil.unpack_archive(question_emb_path, question_emb_dir)

    if progress:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x
    predictions = {}
    for id_, cid in tqdm(q2c.items()):
        q_emb_path = os.path.join(question_emb_dir, '%s.npz' % id_)
        c_emb_path = os.path.join(context_emb_dir, '%s.npz' % cid)
        c_json_path = os.path.join(context_emb_dir, '%s.json' % cid)

        if not os.path.exists(q_emb_path):
            continue

        load = scipy.sparse.load_npz if sparse else np.load
        q_emb = load(q_emb_path)  # shape = [M, d], d is the embedding size.
        c_emb = load(c_emb_path)  # shape = [N, d], d is the embedding size.

        with open(c_json_path, 'r') as fp:
            phrases = json.load(fp)

        if sparse:
            sim = c_emb * q_emb.T
            m = sim.max(1)
            m = np.squeeze(np.array(m.todense()), 1)
        else:
            q_emb = q_emb['arr_0']
            c_emb = c_emb['arr_0']
            sim = np.matmul(c_emb, q_emb.T)
            m = sim.max(1)

        argmax = m.argmax(0)
        predictions[id_] = phrases[argmax]
    
    # Dump piqa_pred
    # with open('test/piqa_pred.json', 'w') as f:
    #     f.write(json.dumps(predictions))

    if context_emb_ext == '.zip':
        shutil.rmtree(context_emb_dir)
    if question_emb_ext == '.zip':
        shutil.rmtree(question_emb_dir)

    return predictions


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('context_emb_dir', help='Context embedding directory')
    parser.add_argument('question_emb_dir', help='Question embedding directory')
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='Whether the embeddings are scipy.sparse or pure numpy.')
    parser.add_argument('--progress', default=False, action='store_true', help='Show progress bar. Requires `tqdm`.')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    q2c = get_q2c(dataset)
    predictions = get_predictions(args.context_emb_dir, args.question_emb_dir, q2c, sparse=args.sparse,
                                  progress=args.progress)
    print(json.dumps(evaluate(dataset, predictions)))
