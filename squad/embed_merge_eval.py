"""
Wrapper script for easy evaluation.

Usage:
    $ python embed_merge_eval.py [--nsml] [--mode] [--large_type]
"""

import subprocess
import argparse
import os
import time, datetime

from pprint import pprint


def run_commands(cmds):
    for cmd_idx, cmd in enumerate(cmds):
        print('Command #{}\n{}'.format(cmd_idx, cmd))
        status = subprocess.call(cmd.split(' '))
        if status != 0:
            print('Failure with exit code: {}'.format(status))
            break


##### For (TF-IDF)=N, (Model)=O, (P/E)=E #####
def run_NOE(nsml, load_dir, iteration, max_eval_par, large_type, no_filter,
            squad_path, large_rand_path, large_tfidf_path, s_question_path,
            context_emb_dir, question_emb_dir, pred_path, draft,
            batch_size, sparse, **kwargs):
    
    c_embed_cmd = ("python main.py analysis --mode embed_context{}{}" +
                   " --load_dir {} --iteration {} --test_path {}" +
                   " --context_emb_dir {} --max_eval_par {}" +
                   " --filter_th {}{} --batch_size {}{}").format(
        ' --cuda' if nsml else '',
        ' --draft' if draft else '',
        load_dir,
        iteration,
        large_rand_path if large_type == 'rand' else large_tfidf_path,
        context_emb_dir,
        max_eval_par,
        0.0 if no_filter else 0.8,
        ' --glove_name glove_squad --preload --num_heads 2 --phrase_filter',
        batch_size,
        ' --sparse' if sparse else ''
    )
    q_embed_cmd = ("python main.py dev --mode embed_question{}{}" +
                   " --load_dir {} --iteration {} --test_path {}"
                   " --question_emb_dir {}{}{}").format(
        ' --cuda' if nsml else '',
        ' --draft' if draft else '',
        load_dir,
        iteration,
        s_question_path,
        question_emb_dir,
        ' --glove_name glove_squad --preload --num_heads 2 --phrase_filter',
        ' --sparse' if sparse else ''
    )
    merge_cmd = "python merge.py {} {} {} {}{}{}".format(
        squad_path,
        context_emb_dir,
        question_emb_dir,
        pred_path,
        ' --q_mat' if not sparse else '',
        ' --sparse' if sparse else ''
    )
    eval_cmd = "python evaluate.py {} {}".format(
        squad_path,
        pred_path
    )
    
    return [c_embed_cmd, q_embed_cmd, merge_cmd, eval_cmd]


##### For (TF-IDF)=Y, (Model)=O, (TF-IDF Mode)=E/P #####
def run_YO(nsml, load_dir, iteration, max_eval_par, large_type, tfidf_weight,
           squad_path, large_rand_path, large_tfidf_path, s_question_path,
           context_emb_dir, question_emb_dir, doc_tfidf_dir, que_tfidf_dir,
           pred_path, draft, tfidf_mode, no_filter, batch_size, sparse, 
           **kwargs):

    c_embed_cmd = ("python main.py analysis --mode embed_context{}{}" +
                   " --load_dir {} --iteration {} --test_path {}" +
                   " --context_emb_dir {} --max_eval_par {}" +
                   " --metadata --filter_th {}{} --batch_size {}{}").format(
        ' --cuda' if nsml else '',
        ' --draft' if draft else '',
        load_dir,
        iteration,
        large_rand_path if large_type == 'rand' else large_tfidf_path,
        context_emb_dir,
        max_eval_par,
        0.0 if no_filter else 0.8,
        ' --glove_name glove_squad --preload --num_heads 2 --phrase_filter',
        batch_size,
        ' --sparse' if sparse else ''
    )
    q_embed_cmd = ("python main.py dev --mode embed_question{}{}" +
                   " --load_dir {} --iteration {} --test_path {}"
                   " --question_emb_dir {}{}{}").format(
        ' --cuda' if nsml else '',
        ' --draft' if draft else '',
        load_dir,
        iteration,
        s_question_path,
        question_emb_dir,
        ' --glove_name glove_squad --preload --num_heads 2 --phrase_filter',
        ' --sparse' if sparse else ''
    )
    merge_cmd = ("python tfidf_merge.py {} {} {} {} {} {}" +
                 " --mode {} --tfidf-weight {}{}{}").format(
        squad_path,
        context_emb_dir,
        doc_tfidf_dir,
        question_emb_dir,
        que_tfidf_dir,
        pred_path,
        tfidf_mode,
        tfidf_weight,
        ' --draft' if draft else '',
        ' --sparse' if sparse else ''
    )
    eval_cmd = "python evaluate.py {} {}".format(
        squad_path,
        pred_path
    )

    return [c_embed_cmd, q_embed_cmd, merge_cmd, eval_cmd]


# Predefined paths (for locals)
data_home = os.path.join(os.path.expanduser('~'), 'data/squad')
CONTEXT_DIR = os.path.join(data_home, 'context_emb_tf')
QUESTION_DIR = os.path.join(data_home, 'question_emb_tf')
DOC_TFIDF_DIR = os.path.join(data_home, 'doc_tfidf')
QUE_TFIDF_DIR = os.path.join(data_home, 'que_tfidf')
SQUAD_PATH = os.path.join(data_home, 'dev-v1.1.json')
S_CONTEXT_PATH = os.path.join(data_home, 'dev-v1.1-context.json')
S_QUESTION_PATH = os.path.join(data_home, 'dev-v1.1-question.json')
LARGE_RAND_PATH = os.path.join(data_home, 'dev-v1.1-large-rand-par100.json')
LARGE_TFIDF_PATH = os.path.join(data_home, 
    'dev-v1.1-large-tfidf-doc30-par100.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embed, merge, then eval')

    # Model
    parser.add_argument('--nsml', default=False, action='store_true',
                        help='Use nsml (default=local)')
    parser.add_argument('--draft', default=False, action='store_true',
                        help='Use draft (default=local)')
    parser.add_argument('--load_dir', type=str, 
                        default='piqateam_minjoon_squad_2_34')
    parser.add_argument('--iteration', type=str, default='35501')
    parser.add_argument('--batch_size', type=str, default=64)
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='Use sparse model (S) (default=false)')

    # Analysis (large setting)
    parser.add_argument('--mode', type=str, default='NOE',
                        help='NOE|YOP|YOE')
    parser.add_argument('--max_eval_par', type=int, default=0)
    parser.add_argument('--large_type', type=str, default='rand',
                        help='rand|tfidf')
    parser.add_argument('--tfidf_weight', type=float, default=1e+0,
                        help='tfidf concat weighting')
    parser.add_argument('--no_filter', default=False, action='store_true',
                        help='No filter (default=use)')

    # Dirs
    parser.add_argument('--context_emb_dir', type=str, default=CONTEXT_DIR)
    parser.add_argument('--question_emb_dir', type=str, default=QUESTION_DIR)
    parser.add_argument('--doc_tfidf_dir', type=str, default=DOC_TFIDF_DIR)
    parser.add_argument('--que_tfidf_dir', type=str, default=QUE_TFIDF_DIR)

    # Paths
    parser.add_argument('--squad_path', type=str, default=SQUAD_PATH)
    parser.add_argument('--s_context_path', type=str, default=S_CONTEXT_PATH)
    parser.add_argument('--s_question_path', type=str, default=S_QUESTION_PATH)
    parser.add_argument('--pred_path', type=str, default='./test_pred.json')
    parser.add_argument('--large_rand_path', type=str, default=LARGE_RAND_PATH)
    parser.add_argument('--large_tfidf_path', type=str, default=LARGE_TFIDF_PATH)
 
    args = parser.parse_args()

    # Change arguments for draft / NSML / sparse
    assert not (args.draft and args.nsml), 'NSML+Draft not supported'
    if args.draft:
        args.load_dir = '/tmp/piqa/squad/save'
        args.iteration = '1'
        args.context_emb_dir = '/tmp/piqa/squad/context_emb'
        args.question_emb_dir = '/tmp/piqa/squad/question_emb'

    if args.nsml:
        from nsml import DATASET_PATH
        nsml_data_home = os.path.join(DATASET_PATH, 'train')
        # args.load_dir = 'piqateam/minjoon_squad_2/34' # (baseline)
        # args.iteration = '35501'
        args.load_dir = 'piqateam/minjoon_squad_2/36' # (sparse)
        args.iteration = '28501'
        args.context_emb_dir = './context_emb'
        args.question_emb_dir = './question_emb'
        args.doc_tfidf_dir = os.path.join(nsml_data_home, 'doc_tfidf')
        args.que_tfidf_dir = os.path.join(nsml_data_home, 'que_tfidf')
        args.squad_path = os.path.join(nsml_data_home, 'dev-v1.1.json')
        args.s_context_path = os.path.join(nsml_data_home,
            'dev-v1.1-context.json')
        args.s_question_path = os.path.join(nsml_data_home,
            'dev-v1.1-question.json')
        args.large_rand_path = os.path.join(nsml_data_home,
            'dev-v1.1-large-rand-par100.json')
        args.large_tfidf_path = os.path.join(nsml_data_home,
            'dev-v1.1-large-tfidf-doc30-par100.json')

    pprint(args.__dict__)

    # Path check
    for key, val in args.__dict__.items():
        if key == 'pred_path': continue
        if 'path' in key:
            assert os.path.exists(val), '{} does not exist'.format(val)

    # Get commands based on the mode
    start = time.time()
    if args.mode == 'NOE':
        cmds = run_NOE(**args.__dict__)
    elif args.mode == 'YOP':
        cmds = run_YO(tfidf_mode='P', **args.__dict__)
    elif args.mode == 'YOE':
        cmds = run_YO(tfidf_mode='E', **args.__dict__)
    else:
        raise NotImplementedError('Not supported mode: {}'.format(args.mode))

    run_commands(cmds)
    print('Time: {}'.format(str(datetime.timedelta(seconds=time.time()-start))))
