import torch
from torch import nn

import dev.model


class Model(dev.Model):

    def get_context(self, **kwargs):
        out = super(Model, self).get_context(**kwargs)

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
            out = super(Model, self).get_context(
                eval_context_char_idxs[:,i,:,:].contiguous(),
                eval_context_glove_idxs[:,i,:].contiguous(),
                eval_context_word_idxs[:,i,:].contiguous(),
                eval_context_elmo_idxs[:,i,:,:].contiguous() \
                if eval_context_elmo_idxs is not None else None
            )

            # Transform to (batch, eval_len, context_len, ..)
            for batch_idx in range(len(out)):
                outs[batch_idx].append(out[batch_idx])
        return outs


class Loss(dev.Loss):
    pass
