import torch
from torch import nn


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
    def __init__(self, word_vocab_size, embed_dim, requires_grad=True):
        super(WordEmbedding, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(word_vocab_size, embed_dim)
        self.embedding.weight.requires_grad = requires_grad

    def forward(self, x):
        device = x.device
        weight_device = self.embedding.weight.device
        x = x.to(weight_device)
        flat_x = x.view(-1, x.size()[-1])
        flat_out = self.embedding(flat_x)
        out = flat_out.view(x.size() + (flat_out.size()[-1],))
        out = out.to(device)
        return out


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
    def __init__(self, char_vocab_size, glove_vocab_size, word_vocab_size, embed_dim, dropout,
                 elmo=False, elmo_options_file=None, elmo_weight_file=None):
        super(Embedding, self).__init__()
        self.char_embedding = CharEmbedding(char_vocab_size, embed_dim)
        self.glove_embedding = WordEmbedding(glove_vocab_size, embed_dim, requires_grad=False)
        self.output_size = 2 * embed_dim
        self.highway1 = Highway(self.output_size, dropout)
        self.highway2 = Highway(self.output_size, dropout)
        if elmo:
            if elmo_options_file is None:
                elmo_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            if elmo_weight_file is None:
                elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            from allennlp.modules.elmo import Elmo
            self.elmo = Elmo(elmo_options_file, elmo_weight_file, 2, dropout=0)
            self.elmo_scale = nn.Parameter(torch.rand(1))
            self.weight = nn.Parameter(torch.rand(2))
            self.softmax = nn.Softmax(dim=0)
            self.sigmoid = nn.Sigmoid()
            self.output_size += self.elmo.get_output_dim()
        else:
            self.elmo = None

    def load_glove(self, glove_emb_mat):
        device = self.glove_embedding.embedding.weight.device
        glove_emb_mat = torch.cat([torch.zeros(2, glove_emb_mat.size()[-1]).to(device), glove_emb_mat], dim=0)
        self.glove_embedding.embedding.weight = torch.nn.Parameter(glove_emb_mat, requires_grad=False)

    def forward(self, cx, gx, x, ex=None):
        cx = self.char_embedding(cx)
        gx = self.glove_embedding(gx)
        output = torch.cat([cx, gx], -1)
        output = self.highway2(self.highway1(output))
        if self.elmo is not None:
            l1, l2 = self.elmo(ex)['elmo_representations']
            weight = self.softmax(self.weight)
            elmo = self.sigmoid(self.elmo_scale) * (l1 * weight[0] + l2 * weight[1])
            output = torch.cat([output, elmo], 2)
        return output


class SelfAtt(nn.Module):
    def __init__(self, input_size, dropout):
        super(SelfAtt, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(input_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_, mask):
        input_ = self.dropout(input_)
        att = self.linear(input_) + mask.unsqueeze(-1)
        att = self.softmax(att)
        output = (input_ * att).sum(1)
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
        query, _ = self.key_lstm(input_)
        key, _ = self.key_lstm(input_)
        att = query.matmul(key.transpose(1, 2)) + mask.unsqueeze(1)
        att = self.softmax(att)
        output = att.matmul(input_)
        return output


class ContextBoundary(nn.Module):
    def __init__(self, embedding, hidden_size, dropout, num_heads):
        super(ContextBoundary, self).__init__()
        self.embedding = embedding
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lstm = torch.nn.LSTM(input_size=self.embedding.output_size,
                                  hidden_size=hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
        self.num_heads = num_heads
        if num_heads > 1:
            for i in range(1, num_heads):
                self.add_module('self_att%d' % i, SelfSeqAtt(hidden_size * 2, hidden_size, dropout))

    def forward(self, cx, gx, x, context_elmo_idxs=None):
        m = ((x == 0).float() * -1e9)
        x = self.embedding(cx, gx, x, ex=context_elmo_idxs)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        if self.num_heads > 1:
            atts = []
            modules = dict(self.named_children())
            for i in range(1, self.num_heads):
                a = modules['self_att%d' % i](x, m)
                atts.append(a)

            x = torch.cat([x] + atts, 2)
        return x


class QuestionBoundary(nn.Module):
    def __init__(self, embedding, hidden_size, dropout, num_heads):
        super(QuestionBoundary, self).__init__()
        self.embedding = embedding
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lstm = torch.nn.LSTM(input_size=self.embedding.output_size,
                                  hidden_size=hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
        self.weight = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.num_heads = num_heads
        for i in range(num_heads):
            self.add_module('att%d' % i, SelfAtt(hidden_size * 2, dropout))

    def forward(self, question_char_idxs, question_glove_idxs, question_word_idxs, question_elmo_idxs=None):
        m = ((question_word_idxs == 0).float() * -1e9)
        q = self.embedding(question_char_idxs, question_glove_idxs, question_word_idxs, ex=question_elmo_idxs)
        q = self.dropout(q)
        q, _ = self.lstm(q)
        atts = []
        modules = dict(self.named_children())
        for i in range(self.num_heads):
            a = modules['att%d' % i](q, m)
            atts.append(a)
        out = torch.cat(atts, 1)
        return out


class PIQA(nn.Module):
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
                 elmo_weight_file=None):
        super(PIQA, self).__init__()
        self.embedding = Embedding(char_vocab_size, glove_vocab_size, word_vocab_size, embed_size, dropout,
                                   elmo=elmo, elmo_options_file=elmo_options_file, elmo_weight_file=elmo_weight_file)
        self.context_start = ContextBoundary(self.embedding, hidden_size, dropout, num_heads)
        self.question_start = QuestionBoundary(self.embedding, hidden_size, dropout, num_heads)
        self.context_end = ContextBoundary(self.embedding, hidden_size, dropout, num_heads)
        self.question_end = QuestionBoundary(self.embedding, hidden_size, dropout, num_heads)
        self.softmax = nn.Softmax(dim=1)
        self.max_ans_len = max_ans_len

    def forward(self,
                context_char_idxs,
                context_glove_idxs,
                context_word_idxs,
                question_char_idxs,
                question_glove_idxs,
                question_word_idxs,
                context_elmo_idxs=None,
                question_elmo_idxs=None,
                **kwargs):
        cx = context_char_idxs
        gx = context_glove_idxs
        x = context_word_idxs
        cq = question_char_idxs
        gq = question_glove_idxs
        q = question_word_idxs
        m = (x == 0).float() * -1e9
        x1 = self.context_start(cx, gx, x, context_elmo_idxs=context_elmo_idxs)
        q1 = self.question_start(cq, gq, q, question_elmo_idxs=question_elmo_idxs)
        x2 = self.context_end(cx, gx, x, context_elmo_idxs=context_elmo_idxs)
        q2 = self.question_end(cq, gq, q, question_elmo_idxs=question_elmo_idxs)

        q1 = q1.view(q1.size()[0], 1, q1.size()[-1])
        q2 = q2.view(q2.size()[0], 1, q2.size()[-1])

        logits1 = torch.sum(x1 * q1, 2) + m
        logits2 = torch.sum(x2 * q2, 2) + m
        prob1 = self.softmax(logits1)
        prob2 = self.softmax(logits2)
        prob = prob1.unsqueeze(2) * prob2.unsqueeze(1)
        mask = (torch.ones(*prob.size()[1:]).triu() - torch.ones(*prob.size()[1:]).triu(self.max_ans_len)).to(prob.device)
        prob *= mask
        _, yp1 = prob.max(2)[0].max(1)
        _, yp2 = prob.max(1)[0].max(1)
        return logits1, logits2, yp1, yp2

    def load_glove(self, glove_emb_mat):
        self.embedding.load_glove(glove_emb_mat)

    def context(self, context_char_idxs, context_glove_idxs, context_word_idxs, **kwargs):
        cx = context_char_idxs
        gx = context_glove_idxs
        x = context_word_idxs
        l = (x > 0).sum(1)
        x1 = self.context_start(cx, gx, x)
        x2 = self.context_end(cx, gx, x)
        out = []
        for lb, x1b, x2b in zip(l, x1, x2):
            pos_list = []
            vec_list = []
            for i in range(lb):
                for j in range(i, min(i+self.max_ans_len, lb)):
                    vec = torch.cat([x1b[i], x2b[j]], 0)
                    pos_list.append((i, j))
                    vec_list.append(vec)
            vec_cat = torch.stack(vec_list, 0)
            out.append((tuple(pos_list), vec_cat))
        return tuple(out)

    def question(self, question_char_idxs, question_glove_idxs, question_word_idxs, **kwargs):
        cq = question_char_idxs
        gq = question_glove_idxs
        q = question_word_idxs
        q1 = self.question_start(cq, gq, q)
        q2 = self.question_end(cq, gq, q)
        out = torch.cat([q1, q2], 1)
        return out


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, answer_word_starts, answer_word_ends, **kwargs):
        loss1 = self.cel(logits1, answer_word_starts[:, 0] - 1)
        loss2 = self.cel(logits2, answer_word_ends[:, 0] - 1)
        loss = loss1 + loss2
        return loss
