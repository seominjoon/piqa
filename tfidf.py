import argparse
import json

import csv
import nltk

from gensim import corpora, models, similarities
from tqdm import tqdm


def tokenize(in_):
    in_ = in_.replace('``', '" ').replace("''", '" ').replace('\t', ' ')
    words = nltk.word_tokenize(in_)
    words = [word.replace('``', '"').replace("''", '"') for word in words]
    return words


def get_phrases_and_documents(context, nbr_len=7, max_ans_len=7, lower=False):
    words = tokenize(context)
    doc_words = [word.lower() for word in words] if lower else words
    phrases = []
    documents = []
    for i in range(len(words)):
        for j in range(i+1, min(len(words), i+max_ans_len)+1):
            phrase = ' '.join(words[i:j])
            document = doc_words[max(0, i-nbr_len):i] + doc_words[j:min(len(words), j+nbr_len)]
            if len(document) == 0:
                continue
            phrases.append(phrase)
            documents.append(document)
    return phrases, documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF-IDF')
    parser.add_argument('data_path')
    parser.add_argument('out_path')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--nbr_len', default=7, type=int)
    parser.add_argument('--max_ans_len', default=7, type=int)
    parser.add_argument('--lower', default=False, action='store_true')
    args = parser.parse_args()

    with open(args.data_path, 'r') as fp:
        examples = list(csv.DictReader(fp))

    out_dict = {}
    for example in tqdm(examples):
        query = tokenize(example['question'])
        phrases, documents = get_phrases_and_documents(example['context'],
                                                       nbr_len=args.nbr_len,
                                                       max_ans_len=args.max_ans_len,
                                                       lower=args.lower)
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        tfidf = models.TfidfModel(corpus)
        index = similarities.MatrixSimilarity(tfidf[corpus])
        sims = index[tfidf[dictionary.doc2bow(query)]]
        phrase = phrases[max(enumerate(sims), key=lambda item: item[1])[0]]
        out_dict[example['id']] = phrase
        if args.draft:
            print(example['question'])
            break

    with open(args.out_path, 'w') as fp:
        json.dump(out_dict, fp)





