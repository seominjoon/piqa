#!/usr/bin/env bash

# Download files into preds
TEST_PATH=preds/tfidf_weighting/baseline/0e+0
SESS=(96) 

cd $TEST_PATH
for i in ${SESS[@]}
do
  nsml download -s /app/preds/agg_pred.json piqateam/squad_piqa_nfs/$i ./
done
cd ../../../../

# Official aggregate and evaluate
python aggregate_pred.py $TEST_PATH
python evaluate.py ~/data/squad/dev-v1.1.json $TEST_PATH/agg_pred.json

: "
# Partial aggregate and evaluate
python aggregate_pred.py $TEST_PATH --with_score
python partial_evaluate.py ~/data/squad/dev-v1.1.json $TEST_PATH/agg_pred.json
"
