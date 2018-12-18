#!/usr/bin/env bash
MAX_CLUSTER=0
TFIDF_WEIGHT=(0e+0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1e+0 3e+0 1e+1 3e+1 1e+2 3e+2 1e+3 3e+3 1e+4 3e+4 1e+5)
N_DOCS=(1 2 3 5 10 20 30)

: "
# Basline dump
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--batch_size 50'
done
"

: "
# Baseline TF-IDF weighting
for i in $(seq 0 $MAX_CLUSTER)
do
  for weight in ${TFIDF_WEIGHT[@]} 
  do 
    echo '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--batch_size 50 --tfidf_weight' $weight
  done
done
"

: "
# Baseline n-docs test
for i in $(seq 0 $MAX_CLUSTER)
do
  for n_docs in ${N_DOCS[@]} 
  do 
    echo '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--batch_size 50 --tfidf_weight 1e-1 --top_n_docs' $n_docs
  done
done

# Baseline merge
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--batch_size 50 --skip_embed'
done
"


: "
# BERT-base dump
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--bert'
done

# BERT-base merge
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--skip_embed --bert'
done
"


: "
# BERT-large dump
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--bert --large'
done

# BERT-large merge
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx' $i '--skip_embed --bert --large'
done
"
