# Phrase-Indexed Question Answering (PIQA)
- This is the official github repository for [Phrase-Indexed Question Answering: A New Challenge for Scalable Document Comprehension][paper] (EMNLP 2018).
- This repository is still in alpha; things might change wildly before versioning starts (currently aiming for Dec 1, 2018). 
- Webpage with leaderboard and submission guideline are coming soon. For now, please consider reproducing the baseline models and running the official evaluation routine (below) to become familiar with the challenge format.
- Much of the work and code is heavily influenced by our former [project][mipsqa] at Google AI.
- For paper-related inquiries, please contact [Minjoon Seo][minjoon] ([@seominjoon][minjoon-github]).
- For code-related inquiries, please create a new issue or contact the admins ([@seominjoon][minjoon-github], [@jhyuklee][jhyuklee-github]).
- For citation, please use:
 ```
@inproceedings{seo2018phrase,
  title={Phrase-Indexed Question Answering: A New Challenge for Scalable Document Comprehension},
  author={Seo, Minjoon and Kwiatkowski, Tom and Parikh, Ankur P and Farhadi, Ali and Hajishirzi, Hannaneh},
  booktitle={EMNLP},
  year={2018}
}
```

## Introduction
We will assume that you have read the [paper][paper], though we will try to recap it here. PIQA challenge is about approaching (existing) extractive question answering tasks via phrase retrieval mechanism (we plan to hold the challenge for several extractive QA datasets in near future, though we currently only support PIQA for [SQuAD 1.1][squad].). This means we need:

1. **document encoder**: enumerates a list of (phrase, vector) pairs from the document,
2. **question encoder**: maps each question to the same vector space, and
3. **retrieval**: retrieves the (phrasal) answer to the question by performing nearest neighbor search on the list. 

While the challenge shares some similarities with document retrieval, a classic problem in information retrieval literature, a key difference is that the phrase representation will need to be *context-based*, which is more challenging than obtaining the embedding by its *content*.

An important aspect of the challenge is the constraint of *independence* between the **document encoder** and the **question encoder**. As we have noted in our paper, most existing models heavily rely on question-dependent representations of the context document. Nevertheless, phrase representations in PIQA need to be completely *independent* of the input question. Not only this makes the challenge quite difficult, but also state-of-the-art models cannot be directly used for the task. Hence we have proposed a few reasonable baseline models as the starting point, which can be found in this repository.

Note that it is also not so straightforward to strictly enforce the constraint on an evaluation platform such as CodaLab. For instance, current SQuAD 1.1 evaluator simply provides the test dataset (both context and question) without answers, and ask the model to output predictions, which are then compared against the answers. This setup is not great for PIQA because we cannot know if the submitted model abides the independence constraint. To resolve this issue, a PIQA submission must consist of the two encoders with explicit independence, and the retrieval is performed on the evaluator side. While it is not as convenient as a vanilla SQuAD submission, we tried to make it as intuitive and easy as possible for the purpose :)

## Tasks

- [Phrase-Indexed SQuAD][pi-squad] (PI-SQuAD)

[paper]: https://arxiv.org/abs/1804.07726
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
[jhyuklee-github]: https://github.com/jhyuklee
[squad-train]: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
[squad-dev]: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
[squad-context]: https://nlp.cs.washington.edu/piqa/squad/dev-v1.1-context.json
[squad-question]: https://nlp.cs.washington.edu/piqa/squad/dev-v1.1-question.json
[elmo]: https://allennlp.org/elmo
[squad]: https://stanford-qa.com
[mipsqa]: https://github.com/google/mipsqa
[pi-squad]: squad/
