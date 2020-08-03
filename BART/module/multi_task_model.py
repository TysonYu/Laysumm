import logging
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from transformers import BartForConditionalGeneration

class multi_task_model(BartForConditionalGeneration):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        ext_label = None,
        sent_ids = None,
        **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Conditional generation example::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."

            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]

            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)

            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        encoder_outputs = outputs[1]
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            # 新增的代码
            sent_linear_layser = nn.Linear(1024, 2).cuda()
            sent_emb = torch.index_select(encoder_outputs, 1, torch.squeeze(sent_ids)).cuda()
            sent_logits = sent_linear_layser(sent_emb)
            sent_loss = loss_fct(torch.squeeze(sent_logits), ext_label.view(-1))
            loss = masked_lm_loss+sent_loss
            
            outputs = (loss,) + outputs
        
        
       
        
        
        return outputs