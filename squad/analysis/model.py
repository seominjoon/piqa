import torch
from torch import nn

import dev.model


class Model(dev.Model):

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

        # Output size: (batch, eval_context + 1, context_len, ...)
        if 'eval_context_char_idxs' in kwargs:
            outs = self.get_eval_context(**kwargs)

            # Last one is the positive context
            for batch_idx in range(len(out)):
                outs[batch_idx].append(out[batch_idx])
            return tuple(outs)
        else:
            return out

    # Use get_context to process 'eval_context'
    def get_eval_context(self, eval_context_char_idxs, eval_context_glove_idxs, eval_context_word_idxs, eval_context_elmo_idxs=None, **kwargs):
        outs = [[] for _ in range(eval_context_word_idxs.size(0))]
        for i in range(eval_context_word_idxs.size(1)):
            # Should not pass 'eval_context_char_idxs' => causes infinite loop
            out = self.get_context(eval_context_char_idxs[:,i,:,:].contiguous(),
                      eval_context_glove_idxs[:,i,:].contiguous(),
                      eval_context_word_idxs[:,i,:].contiguous(),
                      eval_context_elmo_idxs[:,i,:,:].contiguous() \
                      if eval_context_elmo_idxs is not None else None)

            # Transform to (batch, eval_len, context_len, ..)
            for batch_idx in range(len(out)):
                outs[batch_idx].append(out[batch_idx])
        return outs


class Loss(dev.Loss):
    pass
