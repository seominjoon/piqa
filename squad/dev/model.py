import torch
from torch import nn

import baseline.model
from baseline.model import Embedding


class SelfSeqSparse(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super(SelfSeqSparse, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.query_lstm = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
        self.key_lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first=True,
                                bidirectional=True)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, input_, mask):
        input_ = self.dropout(input_)
        key_input = input_
        query_input = input_
        key, _ = self.key_lstm(key_input)
        query, _ = self.key_lstm(query_input)
        sparse = query.matmul(key.transpose(1, 2)) + mask.unsqueeze(1)
        sparse = self.activation(sparse)
        return {'value': sparse, 'key': key, 'query': query}


class ContextBoundary(baseline.model.ContextBoundary):
    def __init__(self, input_size, hidden_size, dropout, num_heads, identity=True, num_layers=1, normalize=False,
                 sparse=False, sparse_activation='relu'):
        super(ContextBoundary, self).__init__(input_size,
                                              hidden_size,
                                              dropout,
                                              num_heads,
                                              identity=identity,
                                              num_layers=num_layers,
                                              normalize=normalize)
        self.sparse = SelfSeqSparse(hidden_size * 2, hidden_size, dropout, sparse_activation) if sparse else None

    def forward(self, x, m):
        modules = dict(self.named_children())
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, _ = modules['lstm%d' % i](x)
        atts = [x] if self.identity else []
        for i in range(self.att_num_heads):
            a = modules['self_att%d' % i](x, m)
            atts.append(a['value'])

        dense = torch.cat(atts, 2)
        sparse = self.sparse(x, m) if self.sparse is not None else None

        if self.normalize:
            dense_norm = 2.0 ** 0.5 * dense.norm(p=2, dim=2, keepdim=True)
            if sparse is None:
                norm = dense_norm
            else:
                sparse_norm = sparse['value'].norm(p=2, dim=2, keepdim=True)
                norm = (dense_norm ** dense_norm + sparse_norm ** sparse_norm) ** 0.5
                sparse['value'] = sparse['value'] / norm
            dense = dense / norm

        return {'dense': dense, 'sparse': sparse['value'] if sparse is not None else None,
                'key': sparse['key'] if sparse is not None else None,
                'query': sparse['query'] if sparse is not None else None}


class QuestionBoundary(ContextBoundary):
    def __init__(self, input_size, hidden_size, dropout, num_heads, num_layers=1, max_pool=False, normalize=False,
                 sparse=False, sparse_activation='relu'):
        super(QuestionBoundary, self).__init__(input_size, hidden_size, dropout, num_heads, identity=False,
                                               num_layers=num_layers,
                                               sparse=sparse, sparse_activation=sparse_activation)
        self.max_pool = max_pool
        self.normalize_ = normalize

    def forward(self, x, m):
        d = super().forward(x, m)
        if self.max_pool:
            dense = d['dense'].max(1)[0]
            sparse = d['sparse'] if d['sparse'] is None else d['sparse'].max(1)[0]
        else:
            dense = d['dense'][:, 0, :]
            sparse = d['sparse'] if d['sparse'] is None else d['sparse'][:, 0, :]

        if self.normalize_:
            dense_norm = 2.0 ** 0.5 * dense.norm(p=2, dim=1, keepdim=True)
            if sparse is None:
                norm = dense_norm
            else:
                sparse_norm = sparse['value'].norm(p=2, dim=1, keepdim=True)
                norm = (dense_norm ** dense_norm + sparse_norm ** sparse_norm) ** 0.5
                sparse['value'] = sparse['value'] / norm
            dense = dense / norm

        return {'dense': dense, 'sparse': sparse}


class Model(baseline.Model):
    def __init__(self,
                 char_vocab_size,
                 glove_vocab_size,
                 word_vocab_size,
                 hidden_size,
                 embed_size,
                 dropout,
                 num_heads,
                 max_ans_len=7,
                 elmo=False,
                 elmo_options_file=None,
                 elmo_weights_file=None,
                 sparse=False,
                 sparse_activation='relu',
                 dense=True,
                 max_pool=False,
                 agg='max',
                 num_layers=1,
                 gen_disc_ratio=0.0,
                 **kwargs):
        super(Model, self).__init__(char_vocab_size,
                                    glove_vocab_size,
                                    word_vocab_size,
                                    hidden_size,
                                    embed_size,
                                    dropout,
                                    num_heads,
                                    max_ans_len=max_ans_len,
                                    elmo=elmo,
                                    elmo_options_file=elmo_options_file,
                                    elmo_weights_file=elmo_weights_file,
                                    max_pool=max_pool,
                                    agg=agg,
                                    num_layers=num_layers,
                                    **kwargs)
        word_size = self.embedding.output_size
        context_input_size = word_size
        question_input_size = word_size
        normalize = self.metric == 'cosine'
        self.sparse = sparse
        self.dense = dense
        self.context_start = ContextBoundary(context_input_size, hidden_size, dropout, num_heads, sparse=sparse,
                                             sparse_activation=sparse_activation, num_layers=num_layers,
                                             normalize=normalize)
        self.context_end = ContextBoundary(context_input_size, hidden_size, dropout, num_heads, sparse=sparse,
                                           sparse_activation=sparse_activation, num_layers=num_layers,
                                           normalize=normalize)
        self.question_start = QuestionBoundary(question_input_size, hidden_size, dropout, num_heads, sparse=sparse,
                                               sparse_activation=sparse_activation, max_pool=max_pool,
                                               normalize=normalize)
        self.question_end = QuestionBoundary(question_input_size, hidden_size, dropout, num_heads, sparse=sparse,
                                             sparse_activation=sparse_activation, max_pool=max_pool,
                                             normalize=normalize)
        self.gen_disc_ratio = gen_disc_ratio
        if gen_disc_ratio > 0.0:
            self.decoder = Decoder(self.embedding.glove_embedding.embedding,
                                   2 * hidden_size * num_heads, num_layers, dropout)

    def forward(self,
                context_char_idxs,
                context_glove_idxs,
                context_word_idxs,
                question_char_idxs,
                question_glove_idxs,
                question_word_idxs,
                context_elmo_idxs=None,
                question_elmo_idxs=None,
                num_samples=None,
                **kwargs):
        q = self.question_embedding(question_char_idxs, question_glove_idxs, question_word_idxs, ex=question_elmo_idxs)
        x = self.context_embedding(context_char_idxs, context_glove_idxs, context_word_idxs, ex=context_elmo_idxs)

        mq = ((question_glove_idxs == 0).float() * -1e9)
        qd1 = self.question_start(q, mq)
        qd2 = self.question_end(q, mq)
        q1, qs1 = qd1['dense'], qd1['sparse']
        q2, qs2 = qd2['dense'], qd2['sparse']
        # print(qs1[0, question_word_idxs[0] > 0])

        mx = (context_glove_idxs == 0).float() * -1e9

        mxq = (context_glove_idxs.unsqueeze(2) == question_glove_idxs.unsqueeze(1)) & (
            context_glove_idxs.unsqueeze(2) > 0)

        hd1 = self.context_start(x, mx)
        hd2 = self.context_end(x, mx)
        x1, xs1 = hd1['dense'], hd1['sparse']
        x2, xs2 = hd2['dense'], hd2['sparse']

        assert self.metric in ('ip', 'cosine', 'l2')

        logits1, logits2 = 0.0, 0.0
        if self.dense:
            logits1 += torch.sum(x1 * q1.unsqueeze(1), 2) + mx
            logits2 += torch.sum(x2 * q2.unsqueeze(1), 2) + mx
        if self.sparse:
            logits1 += (xs1.unsqueeze(-1) * qs1.unsqueeze(1).unsqueeze(1) * mxq.unsqueeze(1).float()).sum([2, 3])
            logits2 += (xs2.unsqueeze(-1) * qs2.unsqueeze(1).unsqueeze(1) * mxq.unsqueeze(1).float()).sum([2, 3])

        if self.metric == 'l2':
            if self.dense:
                logits1 += -0.5 * (torch.sum(x1 * x1, 2) + torch.sum(q1 * q1, 1).unsqueeze(1))
                logits2 += -0.5 * (torch.sum(x2 * x2, 2) + torch.sum(q2 * q2, 1).unsqueeze(1))
            if self.sparse:
                logits1 += -0.5 * (torch.sum(xs1 * xs1, 2) + torch.sum(qs1 * qs1, 1).unsqueeze(1))
                logits2 += -0.5 * (torch.sum(xs2 * xs2, 2) + torch.sum(qs2 * qs2, 1).unsqueeze(1))

        prob1 = self.softmax(logits1)
        prob2 = self.softmax(logits2)
        prob = prob1.unsqueeze(2) * prob2.unsqueeze(1)
        mask = (torch.ones(*prob.size()[1:]).triu() - torch.ones(*prob.size()[1:]).triu(self.max_ans_len)).to(
            prob.device)
        prob *= mask
        _, yp1 = prob.max(2)[0].max(1)
        _, yp2 = prob.max(1)[0].max(1)

        return_ = {'logits1': logits1,
                   'logits2': logits2,
                   'yp1': yp1,
                   'yp2': yp2,
                   'x1': x1,
                   'x2': x2,
                   'q1': q1,
                   'q2': q2,
                   'xs1': xs1,
                   'xs2': xs2,
                   'xsi1': context_glove_idxs,
                   'xsi2': context_glove_idxs,
                   'qs1': qs1,
                   'qs2': qs2,
                   'qsi1': question_glove_idxs,
                   'qsi2': question_glove_idxs}

        if self.gen_disc_ratio > 0.0:
            answer_word_starts = kwargs['answer_word_starts']
            answer_word_ends = kwargs['answer_word_ends']
            eye = torch.eye(context_glove_idxs.size(1)).to(context_glove_idxs.device)
            init1 = torch.embedding(eye, answer_word_starts[:, 0] - 1).unsqueeze(1).matmul(x1).squeeze(1)
            init2 = torch.embedding(eye, answer_word_ends[:, 0] - 1).unsqueeze(1).matmul(x2).squeeze(1)
            decoder_logits1 = self.decoder(init1, question_idxs=question_glove_idxs)
            decoder_logits2 = self.decoder(init2, question_idxs=question_glove_idxs)
            return_['decoder_logits1'] = decoder_logits1
            return_['decoder_logits2'] = decoder_logits2

        return return_

    def get_context(self, context_char_idxs, context_glove_idxs, context_word_idxs, context_elmo_idxs=None, **kwargs):
        l = (context_glove_idxs > 0).sum(1)
        mx = (context_glove_idxs == 0).float() * -1e9
        x = self.context_embedding(context_char_idxs, context_glove_idxs, context_word_idxs, ex=context_elmo_idxs)
        xd1 = self.context_start(x, mx)
        x1, xs1, xsi1 = xd1['dense'], xd1['sparse'], context_glove_idxs
        xd2 = self.context_end(x, mx)
        x2, xs2, xsi2 = xd2['dense'], xd2['sparse'], context_glove_idxs
        out = []
        for k, (lb, x1b, x2b) in enumerate(zip(l, x1, x2)):
            if xs1 is not None:
                xs1b, xs2b = xs1[k], xs2[k]
                xsi1b, xsi2b = xsi1[k], xsi2[k]
                sparse_list = []
                idx_list = []
            pos_list = []
            vec_list = []
            for i in range(lb):
                for j in range(i, min(i + self.max_ans_len, lb)):
                    vec = torch.cat([x1b[i], x2b[j]], 0)
                    pos_list.append((i, j))
                    vec_list.append(vec)
                    if xs1 is not None:
                        sparse = torch.cat([xs1b[i, :lb], xs2b[j, :lb]], 0)
                        idx = torch.cat([xsi1b[:lb], xsi2b[:lb] + 400002], 0)
                        sparse_list.append(sparse)
                        idx_list.append(idx)

            dense = torch.stack(vec_list, 0)
            if xs1 is None:
                sparse = None
            else:
                sparse_cat = None if xs1 is None else torch.stack(sparse_list, 0)
                idx_cat = None if xs1 is None else torch.stack(idx_list, 0)
                sparse = (idx_cat, sparse_cat, 800004)
            out.append((tuple(pos_list), dense, sparse))
        return tuple(out)

    def get_question(self, question_char_idxs, question_glove_idxs, question_word_idxs, question_elmo_idxs=None,
                     **kwargs):
        l = (question_glove_idxs > 0).sum(1)
        mq = ((question_glove_idxs == 0).float() * -1e9)
        q = self.question_embedding(question_char_idxs, question_glove_idxs, question_word_idxs, ex=question_elmo_idxs)
        qd1 = self.question_start(q, mq)
        q1, qs1, qsi1 = qd1['dense'], qd1['sparse'], question_glove_idxs
        qd2 = self.question_end(q, mq)
        q2, qs2, qsi2 = qd2['dense'], qd2['sparse'], question_glove_idxs
        dense_list = list(torch.cat([q1, q2], 1).unsqueeze(1))
        sparse_list = []
        if qs1 is None:
            for lb in l:
                sparse_list.append(None)
        else:
            for lb, val1, idx1, val2, idx2 in zip(l, qs1, qsi1, qs2, qsi2):
                val = torch.cat([val1[:lb], val2[:lb]], 0).unsqueeze(0)
                idx = torch.cat([idx1[:lb], idx2[:lb] + 400002], 0).unsqueeze(0)
                sparse = (idx, val, 800004)
                sparse_list.append(sparse)
        out = tuple(map(tuple, zip(dense_list, sparse_list)))
        return out


class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = embedding
        embed_size = embedding.embedding_dim
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size, embed_size)
        self.start = nn.Parameter(torch.randn(embed_size))

    def forward(self, init, question_idxs=None):
        # Test time; slow for-loop RNN decoding
        if question_idxs is None:
            raise NotImplementedError()
        # Training time; use efficient cudnn RNN
        else:
            targets = self.embedding(question_idxs)
            inputs = torch.cat([self.start.unsqueeze(0).unsqueeze(0).repeat(question_idxs.size(0), 1, 1),
                                targets[:, :-1, :]], 1)
            inputs = self.dropout(inputs)
            outputs, _ = self.gru(inputs, init.unsqueeze(0).repeat(self.num_layers, 1, 1))
            outputs = self.dropout(outputs)
            outputs = self.proj(outputs)

            logits = outputs.view(-1, outputs.size(2)).matmul(self.embedding.weight.t()).view(outputs.size(0),
                                                                                              outputs.size(1), -1)

        return logits


class Loss(baseline.Loss):
    def __init__(self, gen_disc_ratio, loss_ratio_hl, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.gen_disc_ratio = gen_disc_ratio
        self.loss_ratio_hl = loss_ratio_hl

    def forward(self, logits1, logits2, answer_word_starts, answer_word_ends, question_glove_idxs=None,
                decoder_logits1=None, decoder_logits2=None, step=None, **kwargs):
        loss = super(Loss, self).forward(logits1, logits2, answer_word_starts, answer_word_ends)
        if decoder_logits1 is None:
            return loss
        decoder_loss1 = self.cel(decoder_logits1.view(-1, decoder_logits1.size(2)),
                                 question_glove_idxs.view(-1))
        decoder_loss2 = self.cel(decoder_logits2.view(-1, decoder_logits2.size(2)),
                                 question_glove_idxs.view(-1))
        decoder_loss = decoder_loss1 + decoder_loss2
        log2 = torch.log(torch.tensor(2.0).to(decoder_loss.device))
        step = torch.tensor(step).to(decoder_loss.device).float()
        cf = self.gen_disc_ratio * torch.exp(-log2 * step / self.loss_ratio_hl)
        loss += cf * decoder_loss

        return loss
