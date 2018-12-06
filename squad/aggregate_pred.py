import os
import argparse
import json
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregating multiple preds')
    parser.add_argument('--pred_dir', type=str, default='./preds',
                        help='Prediction files directory')
    parser.add_argument('--with_score', default=False, action='store_true')
    args = parser.parse_args()

    # Load prediction files
    preds = []
    for path in os.listdir(args.pred_dir):
        with open(os.path.join(args.pred_dir, path)) as prediction_file:
            preds.append(json.load(prediction_file))
        print('Aggregating {} ...'.format(path))
    
    # Aggregate pred dictionaries
    total_qids = [p.keys() for p in preds]
    total_qids = set([qid for qids in total_qids for qid in qids])
    predictions = {}
    
    for qid in total_qids:
        answers = [p.get(qid, ['', -1e+9]) for p in preds]   
        scores = [a[1] for a in answers]
        best_idx = np.argmax(scores)
        predictions[qid] = (
            answers[best_idx] if args.with_score else answers[best_idx][0]
        )
   
    # Dump as a file
    with open(os.path.join('./new_pred.json'), 'w') as fp:
        json.dump(predictions, fp)
    assert len(predictions) == len(total_qids)
    print('Total {} answers'.format(len(total_qids)))
