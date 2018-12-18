#!/usr/bin/env bash
# Basline dump
: "
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 0 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 1 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 2 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 3 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 4 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 5 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 6 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 7 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 8 --batch_size 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 9 --batch_size 50'
"

# Baseline merge
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 0 --batch_size 50 --skip_embed --embed_session 42'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 1 --batch_size 50 --skip_embed --embed_session 43'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 2 --batch_size 50 --skip_embed --embed_session 44'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 3 --batch_size 50 --skip_embed --embed_session 45'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 4 --batch_size 50 --skip_embed --embed_session 46'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 5 --batch_size 50 --skip_embed --embed_session 47'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 6 --batch_size 50 --skip_embed --embed_session 48'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 7 --batch_size 50 --skip_embed --embed_session 49'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 8 --batch_size 50 --skip_embed --embed_session 50'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 9 --batch_size 50 --skip_embed --embed_session 51'

: "
# BERT-base dump
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 0 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 1 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 2 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 3 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 4 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 5 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 6 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 7 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 8 --bert'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 9 --bert'

# BERT-base merge
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 0 --skip_embed --bert --embed_session 61'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 1 --skip_embed --bert --embed_session 62'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 2 --skip_embed --bert --embed_session 63'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 3 --skip_embed --bert --embed_session 64'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 4 --skip_embed --bert --embed_session 65'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 5 --skip_embed --bert --embed_session 66'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 6 --skip_embed --bert --embed_session 67'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 7 --skip_embed --bert --embed_session 68'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 8 --skip_embed --bert --embed_session 69'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 9 --skip_embed --bert --embed_session 70'

# BERT-large dump
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 0 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 1 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 2 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 3 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 4 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 5 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 6 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 7 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 8 --bert --large'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 9 --bert --large'
"

: "
# BERT-large merge
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 0 --skip_embed --bert --large --embed_session 71'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 1 --skip_embed --bert --large --embed_session 72'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 2 --skip_embed --bert --large --embed_session 73'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 3 --skip_embed --bert --large --embed_session 74'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 4 --skip_embed --bert --large --embed_session 75'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 5 --skip_embed --bert --large --embed_session 76'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 6 --skip_embed --bert --large --embed_session 77'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 7 --skip_embed --bert --large --embed_session 78'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 8 --skip_embed --bert --large --embed_session 79'
nsml run -d squad_piqa_181217 -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx 9 --skip_embed --bert --large --embed_session 80'
"
