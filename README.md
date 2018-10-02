# Phrase-Indexed Question Answering (PIQA)
- This is the official github repository for [Phrase-Indexed Question Answering: A New Challenge for Scalable Document Comprehension][paper] (EMNLP 2018).
- Webpage with leaderboard and submission guideline are coming soon. For now, please consider reproducing the baseline models and running the official evaluation routine (below) to become familiar with the challenge format.
- Much of the work and code are heavily influenced by our former unofficial [repository][mipsqa].
- Please create a new issue on this repository or contact [Minjoon Seo][minjoon] ([@seominjoon][minjoon-github]) for questions and suggestions.

For citation, please use:
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

Note that it is also not so straightforward to strictly enforce the constraint on an evaluation platform such as CodaLab. For instance, current SQuAD 1.1 evaluator simply provides the test dataset (both context and question) without answers, and ask the model to output predictions, which are then compared against the answers. This setup is not great for PIQA because we cannot know if the submitted model abides the independence constraint. To resolve this issue, a PIQA submission must consist of the two encoders with explicit independence, and the retrieval is performed on the evaluator side (See [Submission](#submission) section below). While it is not as convenient as a vanilla SQuAD submission, we tried to make it as intuitive and easy as possible for the purpose :)





## Baseline Models

### 0. Download requirements
Make sure you have Python 3.6 or later. Download and install all requirements by:

```bash
chmod +x download.sh; ./download.sh
```

This will install following python packages:

- `numpy==1.15.2`, `scipy==1.1.0`, `torch==0.4.1`, `nltk==3.3`: Essential packages
- `allenlp==0.6.1`: only if you want to try using [ELMo][elmo]; the installation takes some time.
- `tqdm`, `gensim`: optional.

Download SQuAD v1.1 train and dev set at [`$SQUAD_TRAIN_PATH`][squad-train] and [`$SQUAD_DEV_PATH`][squad-dev], respectively. Also, for official evaluation, download [`$SQUAD_DEV_CONTEXT_PATH`][squad-context] and [`$SQUAD_DEV_QUESTION_PATH`][squad-question]. Note that a simple script `split.py` is used to obtain both files from the original dev dataset.



### 1. Training
In our [paper][paper], we have introduced three baseline models:

For LSTM model:

```bash
python main.py --cuda --train_path $SQUAD_TRAIN_PATH --test_path $SQUAD_DEV_PATH
```

For LSTM+SA model:

```bash
python main.py --cuda --num_heads 2 --train_path $SQUAD_TRAIN_PATH --test_path $SQUAD_DEV_PATH
```

For LSTM+SA+ELMo model (if your GPU does not have 24GB VRAM, ELMo will probably cause OOM so try `--batch_size 32`, which converges a little slowly but still gets a similar accuracy):

```bash
python main.py --cuda --num_heads 2 --elmo --train_path $SQUAD_TRAIN_PATH --test_path $SQUAD_DEV_PATH
```

By default, these commands will output all interesting files (save, report, etc.) to `/tmp/piqa`. You can change the directory with `--output_dir` argument.


### 2. Easy Evaluation
Assuming you trust us, since the baseline code is abiding the independence constraint, let's just try to output the prediction file from a full (context+question) dataset, and evaluate it with the original SQuAD v1.1 evaluator. To do this with LSTM model, simply run:

```bash
python main.py --cuda --mode test --iteration XXXX --test_path $SQUAD_DEV_PATH
```

Where the iteration indicates the step at which the model of interest is saved. Take a look at the standard output during training and pick the one that gives the best performance (which is automatically tracked).
This will output the prediction file at `/tmp/piqa/pred.json`. Now, let's see what the vanilla SQuAD v1.1 evaluator (changed name from `evaluate-v1.1.py` to `evaluate.py`) thinks about it:

```bash
python evaluate.py $SQUAD_DEV_PATH /tmp/piqa/pred.json
```

That was easy! But why is this not an *official evaluation*? Because we had a big assumption in the beginning, that you trust us that our encoders are independent. But who knows?


### 3. Official Evaluation
We need a strict evaluation method that enforces the independence between the encoders. To do so, our (PIQA) evaluator requires three inputs (instead of two). The first input is identical to that of the official evaluator: the path to the test data with the answers. The second and the third correspond to the directories for the phrase embeddings and the question embeddings, respectively. Here is an example directory structure:

```
/tmp/piqa/
+-- context_emb
|   +-- Super_Bowl_50_0.npz
|   +-- Super_Bowl_50_0.json
|   +-- Super_Bowl_50_1.npz
|   +-- Super_Bowl_50_2.json
|   ...
+-- question_emb
|   +-- 56be4eafacb8001400a50302.npz
|   +-- 56d204ade7d4791d00902603.npz
|   ...
```

This looks quite complicated! Let's take a look at one by one.

1. **`.npz` is a numpy/scipy matrix dump**: Each `.npz` file corresponds to a *N-by-d* matrix. If it is a dense matrix, it needs to be saved via `numpy.savez()` method, and if it is a sparse matrix (depending on your need), it needs to be saved via `scipy.sparse.save_npz()` method. Note that `scipy.sparse.save_npz()` is relatively new and old scipy versions do not support it.
2. **each `.npz` in `context_emb` is named after paragraph id**: Here, paragraph id is `'%s_%d' % (article_title, para_idx)`, where `para_idx` indicates the index of the paragraph within the article (starts at `0`). For instance, if the article `Super_Bowl_50` has 35 paragraphs, then it will have `.npz` files until `Super_Bowl_50_34.npz`. 
3. **each `.npz` in `context_emb` is *N* phrase vectors of *d*-dim**: It is up to the submitted model to decide *N* and *d*. For instance, if the paragraph length is 100 words and we enumerate all possible phrases with length <= 7, then we will approximately have *N* = 700. While we will limit the size of `.npz` per word during the submission so one cannot have a very large dense matrix, we will allow sparse matrices, so *d* can be very large in some cases.
4. **`.json` is a list of *N* phrases**: each phrase corresponds to each phrase vector in its corresponding `.npz` file. Of course, one can have duplicate phrases (i.e. several vectors per phrase).
5. **each `.npz` in `question_emb` is named after question id**: Here, question id is the official id in original SQuAD 1.1.
6. **each `.npz` in `question_emb` must be *1*-by-*d* matrix**: Since each question has a single embedding, *N* = 1. Hence the matrix corresponds to the question representation.

Following these rules, one should confirm that `context_emb` contains 4134 files (2067 `.npz` files and 2067 `.json` files, i.e. 2067 paragraphs) and `question_emb` contains 10570 files (one file for each question) for SQuAD v1.1 dev dataset. Hint: `ls context_emb/ | wc -l` gives you the count in the `context_emb` folder. 

In order to output these directories from our model, we run `main.py` with two different arguments for each of document encoder and question encoder.

For document encoder:

```bash
python main.py --cuda --mode embed_context --iteration XXXX --test_path $SQUAD_DEV_CONTEXT_PATH
```

For question encoder:

```bash
python main.py --cuda --mode embed_question --iteration XXXX --test_path $SQUAD_DEV_QUESTION_PATH
```

The encoders will output the embeddings to the default output directory. You can also control the target directories with `--context_emb_dir` and `--question_emb_dir`, respectively. Using uncompressed dense (default) numpy dump for the LSTM model, these directories take about 3 GB of space.  Now one can *officially* evaluate by:

```bash
python piqa_evaluate.py $SQUAD_DEV_PATH /tmp/piqa/context_emb/ /tmp/piqa/question_emb/
```

Note that we currently only support *inner product* for the nearest neighbor search (our baseline model uses inner product as well). We will support L1/L2 distances when the submission opens. Please let us know (create an issue) if you think other measures should be also supported. Note that, however, we try to limit to those that are commonly used for approximate search (so it is unlikely that we will support a multilayer perceptron, because it simply does not scale up).

## Submission
We are coordinating with CodaLab and SQuAD folks to incorporate PIQA evaluation into the CodaLab framework. Submission guideline will be available soon!

[paper]: https://arxiv.org/abs/1804.07726
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
[squad-train]: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
[squad-dev]: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
[squad-context]: https://nlp.cs.washington.edu/piqa/squad/dev-v1.1-context.json
[squad-question]: https://nlp.cs.washington.edu/piqa/squad/dev-v1.1-question.json
[elmo]: https://allennlp.org/elmo
[squad]: https://stanford-qa.com
[mipsqa]: https://github.com/google/mipsqa
