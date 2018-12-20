#!/usr/bin/env bash
MAX_CLUSTER=9
TFIDF_WEIGHT=(1e+1 3e+1 1e+2 3e+2 1e+3 3e+3 1e+4 3e+4 1e+5)  # (0e+0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1e+0 3e+0 1e+1 3e+1 1e+2 3e+2 1e+3 3e+3 1e+4 3e+4 1e+5)
N_DOCS=(1 2 3 10 20 30)

: "
# Basline dump
for i in $(seq 0 $MAX_CLUSTER)
do 
  nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -a '--nsml --embed_c --embed_q --cluster_split 10 --cluster_idx '"$i"' --batch_size 50 --bert --large'
done
"

: "
# Baseline TF-IDF weighting
for weight in ${TFIDF_WEIGHT[@]}
do
  for i in $(seq 0 4) 
  do 
    nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -g 0 -a '--nsml --embed_c --embed_q --cluster_split 5 --cluster_idx '"$i"' --tfidf_weight '"$weight"' --skip_embed' # --bert --large'
  done
done
"

# Baseline n-docs test
BEST_WEIGHT=3e-2
for n_docs in ${N_DOCS[@]} 
do
  for i in $(seq 0 4)
  do 
    nsml run -d squad_piqa_nfs --nfs-output -e embed_merge_eval.py -g 0 -a '--nsml --embed_c --embed_q --cluster_split 5 --cluster_idx '"$i"' --tfidf_weight '"$BEST_WEIGHT"' --top_n_docs '"$n_docs"' --skip_embed --bert --large'
  done
done
: "
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
