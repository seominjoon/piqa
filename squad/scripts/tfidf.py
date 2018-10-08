import argparse
import json

import nltk

from gensim import corpora, models, similarities
from tqdm import tqdm


def load_squad(squad_path, draft=False):
    with open(squad_path, 'r') as fp:
        squad = json.load(fp)
        examples = []
        for article in squad['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    id_ = qa['id']
                    answers, answer_starts, answer_ends = [], [], []
                    for answer in qa['answers']:
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer['text'])
                        answers.append(answer['text'])
                        answer_starts.append(answer_start)
                        answer_ends.append(answer_end)

                    # to avoid csv compatibility issue
                    context = context.replace('\n', '\t')

                    example = {'id': id_,
                               'idx': len(examples),
                               'context': context,
                               'question': question,
                               'answers': answers,
                               'answer_starts': answer_starts,
                               'answer_ends': answer_ends}
                    examples.append(example)
                    if draft and len(examples) == 100:
                        return examples
        return examples


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

    examples = load_squad(args.data_path, draft=args.draft)

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





