"""Official split script for PI-SQuAD v0.1"""


import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Official split script for PI-SQuAD v0.1')
    parser.add_argument('data_path', help='Dataset file path')
    parser.add_argument('context_path', help='Output path for context-only dataset')
    parser.add_argument('question_path', help='Output path for question-only dataset')
    args = parser.parse_args()

    with open(args.data_path, 'r') as fp:
        context = json.load(fp)
    with open(args.data_path, 'r') as fp:
        question = json.load(fp)

    for article in context['data']:
        for para in article['paragraphs']:
            del para['qas']

    for article in question['data']:
        for para in article['paragraphs']:
            del para['context']
            for qa in para['qas']:
                if 'answers' in qa:
                    del qa['answers']

    with open(args.context_path, 'w') as fp:
        json.dump(context, fp)

    with open(args.question_path, 'w') as fp:
        json.dump(question, fp)
