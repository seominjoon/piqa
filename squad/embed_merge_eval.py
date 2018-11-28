"""
Wrapper script for easy evaluation.

Usage:
    $ python embed_merge_eval.py [--nsml] [--mode] [--large_type]
"""

import subprocess
import argparse
import os


def run_commands(cmds):
    for cmd_idx, cmd in enumerate(cmds):
        print('Command #{}\n{}'.format(cmd_idx, cmd))
        status = subprocess.call(cmd.split(' '))
        if status != 0:
            exit(status)


##### For (TF-IDF)=N, (Model)=O, (P/E)=E #####
def run_NOE(nsml, load_dir, iteration, max_eval_par, large_type,
            squad_path, large_rand_path, large_tfidf_path, s_question_path,
            context_emb_dir, question_emb_dir, pred_path, **kwargs):
    
    c_embed_cmd = ("python main.py analysis --mode embed_context {}" +
                   " --load_dir {} --iteration {} --test_path {}" +
                   " --context_emb_dir {} --max_eval_par {}").format(
        '--cuda' if nsml else '--draft',
        load_dir,
        iteration,
        large_rand_path if large_type == 'rand' else large_tfidf_path,
        context_emb_dir,
        max_eval_par
    )
    q_embed_cmd = ("python main.py baseline --mode embed_question {}" +
                   " --load_dir {} --iteration {} --test_path {}"
                   " --question_emb_dir {}").format(
        '--cuda' if nsml else '--draft',
        load_dir,
        iteration,
        s_question_path,
        question_emb_dir
    )
    merge_cmd = "python merge.py {} {} {} {}".format(
        squad_path,
        context_emb_dir,
        question_emb_dir,
        pred_path
    )
    eval_cmd = "python evaluate.py {} {}".format(
        squad_path,
        pred_path
    )
    
    return [c_embed_cmd, q_embed_cmd, merge_cmd, eval_cmd]


##### For (TF-IDF)=Y, (Model)=O, (P/E)=P #####
def run_YOP(**kwargs):
    raise NotImplementedError()


##### For (TF-IDF)=Y, (Model)=O, (P/E)=E #####
def run_YOE(nsml, load_dir, iteration, max_eval_par, large_type,
            squad_path, large_rand_path, large_tfidf_path, s_question_path,
            context_emb_dir, question_emb_dir, doc_tfidf_dir, que_tfidf_dir,
            pred_path, **kwargs):

    c_embed_cmd = ("python main.py analysis --mode embed_context {}" +
                   " --load_dir {} --iteration {} --test_path {}" +
                   " --context_emb_dir {} --max_eval_par {} --metadata").format(
        '--cuda' if nsml else '--draft',
        load_dir,
        iteration,
        large_rand_path if large_type == 'rand' else large_tfidf_path,
        context_emb_dir,
        max_eval_par
    )
    q_embed_cmd = ("python main.py baseline --mode embed_question {}" +
                   " --load_dir {} --iteration {} --test_path {}"
                   " --question_emb_dir {}").format(
        '--cuda' if nsml else '--draft',
        load_dir,
        iteration,
        s_question_path,
        question_emb_dir
    )
    merge_cmd = "python tfidf_merge.py {} {} {} {} {} {}{}".format(
        squad_path,
        context_emb_dir,
        doc_tfidf_dir,
        question_emb_dir,
        que_tfidf_dir,
        pred_path,
        '--draft' if not nsml else ''
    )
    eval_cmd = "python evaluate.py {} {}".format(
        squad_path,
        pred_path
    )

    return [c_embed_cmd, q_embed_cmd, merge_cmd, eval_cmd]


# Predefined paths (for locals)
data_home = os.path.join(os.path.expanduser('~'), 'data/squad')
CONTEXT_DIR = '/tmp/piqa/squad/context_emb/'
QUESTION_DIR = '/tmp/piqa/squad/question_emb/'
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
    parser.add_argument('--load_dir', type=str, default='/tmp/piqa/squad/save')
    parser.add_argument('--iteration', type=str, default='1')

    # Analysis (large setting)
    parser.add_argument('--mode', type=str, default='NOE',
                        help='NOE|YOP|YOE')
    parser.add_argument('--max_eval_par', type=int, default=0)
    parser.add_argument('--large_type', type=str, default='rand',
                        help='rand|tfidf')

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
    print(args.__dict__)

    # Change arguments for NSML 
    if args.nsml:
        nsml_data_home = '../data/squad_piqa_181128/train'
        args.load_dir = 'piqateam/minjoon_squad_2/37'
        args.iteration = '35501'
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
            'dev-v1.1-large-tfidf-par100.json')

    # Path check
    for key, val in args.__dict__.items():
        if key == 'pred_path': continue
        if 'path' in key:
            assert os.path.exists(val), '{} does not exist'.format(val)

    # Get commands based on the mode
    if args.mode == 'NOE':
        cmds = run_NOE(**args.__dict__)
    elif args.mode == 'YOP':
        cmds = run_YOP(**args.__dict__)
    elif args.mode == 'YOE':
        cmds = run_YOE(**args.__dict__)
    else:
        raise NotImplementedError('Not supported mode: {}'.format(args.mode))

    run_commands(cmds)
