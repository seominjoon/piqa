import argparse
import json
import random
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF-IDF')
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    parser.add_argument('--num_per_para', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    with open(args.in_path, 'r') as fp:
        in_ = json.load(fp)

    num_articles = len(in_['data'])
    for ai, article in enumerate(in_['data']):
        other_articles = in_['data'][:ai] + in_['data'][ai+1:]
        for para in article['paragraphs']:
            for i in range(args.num_per_para):
                other_article = random.choice(other_articles)
                other_para = random.choice(other_article['paragraphs'])
                other_qa = copy.deepcopy(random.choice(other_para['qas']))
                other_qa['answers'] = []
                para['qas'].append(other_qa)

    with open(args.out_path, 'w') as fp:
        json.dump(in_, fp)