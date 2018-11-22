import subprocess

##### For baseline testing #####
'''
# Run embed_context
return_code = subprocess.call("python main.py baseline --cuda --mode embed_context --load_dir KR62034/squad_jnhk/8 --iteration 24001 --test_path ../data/squad_piqa/train/dev-v1.1-context.json --context_emb_dir ./context_emb".split(' '))
print('Return code: {}'.format(return_code))

if return_code == 0:
    # Run embed_question
    return_code = subprocess.call("python main.py baseline --cuda --mode embed_question --load_dir KR62034/squad_jnhk/8 --iteration 24001 --test_path ../data/squad_piqa/train/dev-v1.1-question.json --question_emb_dir ./question_emb".split(' '))
else:
    exit(1)
print('Return code: {}'.format(return_code))

if return_code == 0:
    # Run merge.py
    return_code = subprocess.call("python merge.py ../data/squad_piqa/train/dev-v1.1.json ./context_emb ./question_emb ./piqa_pred.json".split(' '))
else:
    exit(1)
print('Return code: {}'.format(return_code))

if return_code == 0:
    # Run evaluate.py
    return_code = subprocess.call("python evaluate.py ../data/squad_piqa/train/dev-v1.1.json ./piqa_pred.json".split(' '))
else:
    exit(1)
print('Return code: {}'.format(return_code))
'''

##### For analysis testing (with baseline) #####
'''
'''
# Run embed_context
return_code = subprocess.call("python main.py analysis --max_eval_par 5 --cuda --mode embed_context --load_dir KR62034/squad_piqa/23 --iteration 18501 --test_path ../data/squad_piqa_analysis/train/dev-v1.1-top100-eval-par10.json --context_emb_dir ./context_emb".split(' '))
print('Return code: {}'.format(return_code))

if return_code == 0:
    # Run embed_question
    return_code = subprocess.call("python main.py baseline --cuda --mode embed_question --load_dir KR62034/squad_piqa/23 --iteration 18501 --test_path ../data/squad_piqa_analysis/train/dev-v1.1-question.json --question_emb_dir ./question_emb".split(' '))
else:
    exit(1)
print('Return code: {}'.format(return_code))

if return_code == 0:
    # Run merge.py
    return_code = subprocess.call("python merge.py ../data/squad_piqa_analysis/train/dev-v1.1.json ./context_emb ./question_emb ./piqa_pred.json".split(' '))
else:
    exit(1)
print('Return code: {}'.format(return_code))

if return_code == 0:
    # Run evaluate.py
    return_code = subprocess.call("python evaluate.py ../data/squad_piqa_analysis/train/dev-v1.1.json ./piqa_pred.json".split(' '))
else:
    exit(1)
print('Return code: {}'.format(return_code))
