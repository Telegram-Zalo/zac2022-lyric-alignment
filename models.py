from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import (
    Wav2Vec2Processor,
)

from dataclasses import dataclass
from transformers.utils import ModelOutput


@dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
            loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                    Language modeling loss (for next-token prediction).
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
                    Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                    Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
                    one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                    Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                    sequence_length)`.

                    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                    heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # boundary_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ForCTCV2(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)

    def resize_lm_head(self, new_num_tokens=107):
        old_lm_head = self.lm_head
        # Build new lm head
        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size()
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(
            old_lm_head.weight.device, dtype=old_lm_head.weight.dtype
        )
        self._init_weights(new_lm_head)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
            :num_tokens_to_copy, :
        ]
        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
            :num_tokens_to_copy
        ]
        self.lm_head = new_lm_head

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                        Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
                        the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                        All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                        config.vocab_size - 1]`.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_logits = self.lm_head(self.dropout(encoder_outputs[0]))

        loss = None
        if labels is not None:
            # Encoder loss
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                encoder_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            with torch.backends.cudnn.flags(enabled=False):
                encoder_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            # Sum loss
            loss = encoder_loss

        if not return_dict:
            output = (encoder_logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=encoder_logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
