"""
Wrapper script for easy evaluation.

Usage:
    $ python embed_merge_eval.py [--nsml]
"""

import subprocess
import argparse
import os
import time, datetime

from pprint import pprint


def run_commands(cmds):
    start = time.time()
    for cmd_idx, cmd in enumerate(cmds):
        print('\nCommand #{}\n{}'.format(cmd_idx, cmd))
        status = subprocess.call(cmd.split(' '))
        if status != 0:
            print('Failure with exit code: {}'.format(status))
            # return str(datetime.timedelta(seconds=time.time()-start))
    return str(datetime.timedelta(seconds=time.time()-start))


def embed_question(nsml, draft, load_dir, iteration,
                   question_path, question_emb_dir,
                   sparse, **kwargs):

    q_embed_cmd = ("python main.py dev --mode embed_question{}{}{}" +
                   " --load_dir {} --iteration {} --test_path {}"
                   " --question_emb_dir {}" +
                   " --glove_name glove_squad --preload" +
                   " --num_heads 2 --phrase_filter").format(
        ' --cuda' if nsml else '',
        ' --draft' if draft else '',
        ' --sparse' if sparse else '',
        load_dir,
        iteration,
        question_path,
        question_emb_dir
    )

    return [q_embed_cmd]


def embed_context(nsml, draft, load_dir, iteration,
                  context_paths, context_emb_dirs, pred_paths,
                  no_filter, sparse, batch_size, **kwargs):

    cmds = []
    for context_path, context_emb_dir, pred_path in zip(
        context_paths, context_emb_dirs, pred_paths):
        c_embed_cmd = ("python main.py dev --mode embed_context{}{}{}" +
                       " --load_dir {} --iteration {} --test_path {}" +
                       " --context_emb_dir {}" +
                       " --filter_th {} --batch_size {}" +
                       " --glove_name glove_squad --preload" +
                       " --num_heads 2 --phrase_filter").format(
            ' --cuda' if nsml else '',
            ' --draft' if draft else '',
            ' --sparse' if sparse else '',
            load_dir,
            iteration,
            context_path,
            context_emb_dir,
            0.0 if no_filter else 0.8,
            batch_size
        )
        cmds.append(c_embed_cmd)
        merge_cmd = merge(
            nsml=nsml,
            draft=draft,
            sparse=sparse,
            context_path=context_path,
            context_emb_dir=context_emb_dir,
            pred_path=pred_path,
            **kwargs
        )
        cmds += merge_cmd

    return cmds


def merge(nsml, draft, sparse, context_path,
          d2q_path, context_emb_dir, question_emb_dir, pred_path,
          tfidf_weight, top_n_docs, **kwargs):
    merge_cmd = ("python tfidf_merge.py {} {} {} {} {}" +
                 " --tfidf-weight {} --top-n-docs {}{}").format(
        context_emb_dir,
        question_emb_dir,
        d2q_path,
        context_path,
        pred_path,
        tfidf_weight,
        top_n_docs,
        ' --sparse' if sparse else ''
    )
    return [merge_cmd]


def aggregate(squad_path, pred_dir, **kwargs):
    agg_cmd = "python aggregate_pred.py --pred_dir {} --with_score".format(
        pred_dir
    )
    eval_cmd = "python partial_evaluate.py {} new_pred.json".format(
        squad_path
    )
    return [agg_cmd, eval_cmd]
    

# Predefined paths (for locals)
data_home = os.path.join(os.path.expanduser('~'), 'data/squad')
CONTEXT_BASE = os.path.join(data_home, 'context_emb_30')
QUESTION_DIR = os.path.join(data_home, 'question_emb_30')
SQUAD_PATH = os.path.join(data_home, 'dev-v1.1.json')
D2Q_PATH = os.path.join(data_home, 'd2q_30.json')
QUESTION_PATH = os.path.join(data_home, 'dev-v1.1-question.json')
CONTEXT_PATHS = [os.path.join(data_home,
    'dev_contexts/top30/dev-v1.1-top30docs-{}.json'.format(k)) 
    for k in range(100)]
PREDICTION_DIR = './preds'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embed, merge, then eval')

    # Load / Run setting
    parser.add_argument('--nsml', default=False, action='store_true',
                        help='Use nsml (default=local)')
    parser.add_argument('--draft', default=False, action='store_true',
                        help='Use draft (default=local)')
    parser.add_argument('--load_dir', type=str, 
                        default='piqateam_minjoon_squad_2_34')
                        # default='piqateam_minjoon_squad_2_36')
    parser.add_argument('--iteration', type=str,
                        default='35501')
                        # default='28501')

    # NSML reserved
    parser.add_argument('--mode', type=str, default='fork')
    parser.add_argument('--pause', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default=None)

    # Mode
    parser.add_argument('--embed_c', default=False, action='store_true')
    parser.add_argument('--embed_q', default=False, action='store_true')

    # Controllable arguments
    parser.add_argument('--tfidf_weight', type=float, default=0.0,
                        help='tfidf concat weighting')
    parser.add_argument('--top_n_docs', type=int, default=5,
                        help='Number of documents to eval')
    parser.add_argument('--no_filter', default=False, action='store_true',
                        help='No filter (default=use)')
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='Use sparse model (S) (default=false)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cluster_idx', type=int, default=5, help='0~')
    parser.add_argument('--cluster_split', type=int, default=50)

    # Dirs
    parser.add_argument('--context_emb_base', type=str, default=CONTEXT_BASE)
    parser.add_argument('--question_emb_dir', type=str, default=QUESTION_DIR)
    parser.add_argument('--pred_dir', type=str, default=PREDICTION_DIR)

    # Paths
    parser.add_argument('--squad_path', type=str, default=SQUAD_PATH)
    parser.add_argument('--d2q_path', type=str, default=D2Q_PATH)
    parser.add_argument('--context_paths', type=list, default=CONTEXT_PATHS)
    parser.add_argument('--question_path', type=str, default=QUESTION_PATH)
 
    args = parser.parse_args()

    # Change arguments for draft / NSML / sparse
    assert not (args.draft and args.nsml), 'NSML+Draft not supported'
    # TODO: debug for draft (not working probably)
    if args.draft:
        args.load_dir = '/tmp/piqa/squad/save'
        args.iteration = '1'
        args.context_emb_base = '/tmp/piqa/squad/context_emb'
        args.question_emb_dir = '/tmp/piqa/squad/question_emb'

    if args.nsml:
        from nsml import DATASET_PATH
        nsml_data_home = os.path.join(DATASET_PATH, 'train')
        args.load_dir = 'piqateam/minjoon_squad_2/34' # (baseline)
        args.iteration = '35501'
        # args.load_dir = 'piqateam/minjoon_squad_2/36' # (sparse)
        # args.iteration = '28501'
        args.context_emb_base = './context_emb'
        args.question_emb_dir = './question_emb'
        args.squad_path = os.path.join(nsml_data_home, 'dev-v1.1.json')
        args.d2q_path = os.path.join(nsml_data_home, 'd2q_30.json')
        args.context_paths = [os.path.join(nsml_data_home,
            'top30/dev-v1.1-top30docs-{}.json'.format(k)) 
            for k in range(100)]
        args.question_path = os.path.join(nsml_data_home,
            'dev-v1.1-question.json')

    # Change context_paths according to cluster idx, then edit dirs/paths
    num_docs = len(args.context_paths) // args.cluster_split
    args.context_paths = args.context_paths[
        num_docs*args.cluster_idx:num_docs*(args.cluster_idx+1)
    ]
    context_emb_dirs = []
    pred_paths = []
    for path_idx in range(len(args.context_paths)):
        save_idx = path_idx + num_docs * args.cluster_idx
        context_emb_dirs.append(
            os.path.join(args.context_emb_base, str(save_idx))
        )
        pred_paths.append(
            os.path.join(args.pred_dir, 'pred_{}.json'.format(save_idx))
        )
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
    pprint(args.__dict__)

    # Path check
    for key, val in args.__dict__.items():
        if 'paths' in key:
            for path in val:
                assert os.path.exists(path), '{} does not exist'.format(path)
        elif 'path' in key:
            assert os.path.exists(val), '{} does not exist'.format(val)

    # Get commands based on the mode
    if args.embed_q:
        cmds = embed_question(**args.__dict__)
        elapsed = run_commands(cmds)
        print('question embed: {}'.format(elapsed))

    if args.embed_c:
        cmds = embed_context(
            context_emb_dirs=context_emb_dirs,
            pred_paths=pred_paths,
            **args.__dict__
        )
        cmds += aggregate(**args.__dict__)
        elapsed = run_commands(cmds)
        print('context embed + merge + aggregate: {}'.format(elapsed))

