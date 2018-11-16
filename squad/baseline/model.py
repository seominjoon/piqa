import torch
from torch import nn

import base


class CharEmbedding(nn.Module):
    def __init__(self, char_vocab_size, embed_dim):
        super(CharEmbedding, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(char_vocab_size, embed_dim)

    def forward(self, x):
        flat_x = x.view(-1, x.size()[-1])
        flat_out = self.embedding(flat_x)
        out = flat_out.view(x.size() + (flat_out.size()[-1],))
        out, _ = torch.max(out, -2)
        return out


class WordEmbedding(nn.Module):
    def __init__(self, word_vocab_size=None, embed_dim=None, requires_grad=True, cpu=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(word_vocab_size, embed_dim)
        self.embedding.weight.requires_grad = requires_grad
        self._cpu = cpu

    def forward(self, x):
        device = x.device
        weight_device = self.embedding.weight.device
        x = x.to(weight_device)
        flat_x = x.view(-1, x.size()[-1])
        flat_out = self.embedding(flat_x)
        out = flat_out.view(x.size() + (flat_out.size()[-1],))
        out = out.to(device)
        return out

    def to(self, device):
        return self if self._cpu else super().to(device)


class Highway(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Highway, self).__init__()
        self.input_linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.gate_linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_):
        input_ = self.dropout(input_)
        output = self.relu(self.input_linear(input_))
        gate = self.sigmoid(self.gate_linear(input_))
        output = input_ * gate + output * (1.0 - gate)
        return output


class Embedding(nn.Module):
    def __init__(self, char_vocab_size, glove_vocab_size, word_vocab_size, embed_dim, dropout, elmo=False,
                 glove_cpu=False):
        super(Embedding, self).__init__()
        self.word_embedding = WordEmbedding(word_vocab_size, embed_dim)
        self.char_embedding = CharEmbedding(char_vocab_size, embed_dim)
        self.glove_embedding = WordEmbedding(glove_vocab_size, embed_dim, requires_grad=False, cpu=glove_cpu)
        self.output_size = 2 * embed_dim
        self.highway1 = Highway(self.output_size, dropout)
        self.highway2 = Highway(self.output_size, dropout)
        self.use_elmo = elmo
        self.elmo = None
        if self.use_elmo:
            self.output_size += 1024

    def load_glove(self, glove_emb_mat):
        device = self.glove_embedding.embedding.weight.device
        glove_emb_mat = glove_emb_mat.to(device)
        glove_emb_mat = torch.cat([torch.zeros(2, glove_emb_mat.size()[-1]).to(device), glove_emb_mat], dim=0)
        self.glove_embedding.embedding.weight = torch.nn.Parameter(glove_emb_mat, requires_grad=False)

    def load_elmo(self, elmo_options_file, elmo_weights_file):
        device = self.word_embedding.embedding.weight.device
        from allennlp.modules.elmo import Elmo
        self.elmo = Elmo(elmo_options_file, elmo_weights_file, 1, dropout=0).to(device)

    def init(self, processed_metadata):
        self.load_glove(processed_metadata['glove_emb_mat'])
        if self.use_elmo:
            self.load_elmo(processed_metadata['elmo_options_file'], processed_metadata['elmo_weights_file'])

    def forward(self, cx, gx, x, ex=None):
        cx = self.char_embedding(cx)
        gx = self.glove_embedding(gx)
        output = torch.cat([cx, gx], -1)
        output = self.highway2(self.highway1(output))
        if self.use_elmo:
            elmo, = self.elmo(ex)['elmo_representations']
            output = torch.cat([output, elmo], 2)
        return output


class SelfSeqAtt(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(SelfSeqAtt, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.query_lstm = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
        self.key_lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first=True,
                                bidirectional=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_, mask):
        input_ = self.dropout(input_)
        key_input = input_
        query_input = input_
        key, _ = self.key_lstm(key_input)
        query, _ = self.key_lstm(query_input)
        att = query.matmul(key.transpose(1, 2)) + mask.unsqueeze(1)
        att = self.softmax(att)
        output = att.matmul(input_)
        return {'value': output, 'key': key, 'query': query}


class ContextBoundary(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_heads, identity=True, num_layers=1, normalize=False):
        super(ContextBoundary, self).__init__()
        assert num_heads >= 1, num_heads
        self.normalize = normalize
        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.add_module('lstm%d' % i, torch.nn.LSTM(input_size=input_size,
                                                        hidden_size=hidden_size,
                                                        batch_first=True,
                                                        bidirectional=True))
        self.num_heads = num_heads
        self.identity = identity
        self.att_num_heads = num_heads - 1 if identity else num_heads
        for i in range(self.att_num_heads):
            self.add_module('self_att%d' % i,
                            SelfSeqAtt(hidden_size * 2, hidden_size, dropout))

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
        if self.normalize:
            dense = dense / dense.norm(p=2, dim=2, keepdim=True)
        return {'dense': dense}


class QuestionBoundary(ContextBoundary):
    def __init__(self, input_size, hidden_size, dropout, num_heads, num_layers=1,
                 max_pool=False, normalize=False):
        # No need to normalize question
        super(QuestionBoundary, self).__init__(input_size, hidden_size, dropout, num_heads, identity=False,
                                               num_layers=num_layers)
        self.max_pool = max_pool
        self.normalize_ = normalize

    def forward(self, x, m):
        d = super().forward(x, m)
        if self.max_pool:
            dense = d['dense'].max(1)[0]
        else:
            dense = d['dense'][:, 0, :]

        if self.normalize_:
            dense = dense / dense.norm(p=2, dim=1, keepdim=True)
        return {'dense': dense}


class Model(base.Model):
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
                 max_pool=False,
                 num_layers=1,
                 glove_cpu=False,
                 metric='ip',
                 **kwargs):
        super(Model, self).__init__()
        self.embedding = Embedding(char_vocab_size, glove_vocab_size, word_vocab_size, embed_size, dropout,
                                   elmo=elmo, glove_cpu=glove_cpu)
        self.context_embedding = self.embedding
        self.question_embedding = self.embedding
        word_size = self.embedding.output_size
        context_input_size = word_size
        question_input_size = word_size
        normalize = metric == 'cosine'
        self.context_start = ContextBoundary(context_input_size, hidden_size, dropout, num_heads, num_layers=num_layers,
                                             normalize=normalize)
        self.context_end = ContextBoundary(context_input_size, hidden_size, dropout, num_heads, num_layers=num_layers,
                                           normalize=normalize)
        self.question_start = QuestionBoundary(question_input_size, hidden_size, dropout, num_heads, max_pool=max_pool,
                                               normalize=normalize)
        self.question_end = QuestionBoundary(question_input_size, hidden_size, dropout, num_heads, max_pool=max_pool,
                                             normalize=normalize)
        self.softmax = nn.Softmax(dim=1)
        self.max_ans_len = max_ans_len
        self.linear = nn.Linear(word_size, 1)
        self.metric = metric

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
        q1 = qd1['dense']
        q2 = qd2['dense']
        # print(qs1[0, question_word_idxs[0] > 0])

        mx = (context_glove_idxs == 0).float() * -1e9

        hd1 = self.context_start(x, mx)
        hd2 = self.context_end(x, mx)
        x1 = hd1['dense']
        x2 = hd2['dense']

        assert self.metric in ('ip', 'cosine', 'l2')

        logits1 = torch.sum(x1 * q1.unsqueeze(1), 2) + mx
        logits2 = torch.sum(x2 * q2.unsqueeze(1), 2) + mx

        if self.metric == 'l2':
            logits1 += -0.5 * (torch.sum(x1 * x1, 2) + torch.sum(q1 * q1, 1).unsqueeze(1))
            logits2 += -0.5 * (torch.sum(x2 * x2, 2) + torch.sum(q2 * q2, 1).unsqueeze(1))

        prob1 = self.softmax(logits1)
        prob2 = self.softmax(logits2)
        prob = prob1.unsqueeze(2) * prob2.unsqueeze(1)
        mask = (torch.ones(*prob.size()[1:]).triu() - torch.ones(*prob.size()[1:]).triu(self.max_ans_len)).to(
            prob.device)
        prob *= mask
        _, yp1 = prob.max(2)[0].max(1)
        _, yp2 = prob.max(1)[0].max(1)

        return {'logits1': logits1,
                'logits2': logits2,
                'yp1': yp1,
                'yp2': yp2,
                'x1': x1,
                'x2': x2,
                'q1': q1,
                'q2': q2}

    def init(self, processed_metadata):
        self.embedding.init(processed_metadata)

    def get_context(self, context_char_idxs, context_glove_idxs, context_word_idxs, context_elmo_idxs=None, **kwargs):
        l = (context_glove_idxs > 0).sum(1)
        mx = (context_glove_idxs == 0).float() * -1e9
        x = self.context_embedding(context_char_idxs, context_glove_idxs, context_word_idxs, ex=context_elmo_idxs)
        xd1 = self.context_start(x, mx)
        x1 = xd1['dense']
        xd2 = self.context_end(x, mx)
        x2 = xd2['dense']
        out = []
        for k, (lb, x1b, x2b) in enumerate(zip(l, x1, x2)):
            pos_list = []
            vec_list = []
            for i in range(lb):
                for j in range(i, min(i + self.max_ans_len, lb)):
                    vec = torch.cat([x1b[i], x2b[j]], 0)
                    pos_list.append((i, j))
                    vec_list.append(vec)

            dense = torch.stack(vec_list, 0)
            out.append((tuple(pos_list), dense))
        return tuple(out)

    def get_question(self, question_char_idxs, question_glove_idxs, question_word_idxs, question_elmo_idxs=None,
                     **kwargs):
        mq = ((question_glove_idxs == 0).float() * -1e9)
        q = self.question_embedding(question_char_idxs, question_glove_idxs, question_word_idxs, ex=question_elmo_idxs)
        qd1 = self.question_start(q, mq)
        q1 = qd1['dense']
        qd2 = self.question_end(q, mq)
        q2 = qd2['dense']
        out = list(torch.cat([q1, q2], 1).unsqueeze(1))
        return out


class Loss(base.Loss):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, answer_word_starts, answer_word_ends, **kwargs):
        answer_word_starts -= 1
        answer_word_ends -= 1
        loss1 = self.cel(logits1, answer_word_starts[:, 0])
        loss2 = self.cel(logits2, answer_word_ends[:, 0])
        loss = loss1 + loss2
        return loss
