#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoModel
from modules.net_models import MeanPooling, AttentionPooling

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Dict, List, Tuple, Union
@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    embeds: Optional[torch.FloatTensor] = None
    neuron_to_feat: Optional[Dict[int, List[Tuple[int, float]]]] = None


class TransformerVARegressor(nn.Module):
    """
    BERT-style regressor for predicting Valence and Arousal scores.

    Args:
        model_name: HuggingFace model name, e.g. "bert-base-multilingual-cased".
        dropout: Dropout rate before the regression head.
    """

    def __init__(
        self, model_name: str = "bert-base-multilingual-cased", dropout: float = 0.1, pool_type="cls", use_flash_attn=False,
    ) -> None:
        super().__init__()
        if use_flash_attn:
            self.backbone = AutoModel.from_pretrained(model_name, attn_implementation="flash_attention_2")
        else:
            self.backbone = AutoModel.from_pretrained(model_name)#, attn_implementation="flash_attention_2")
        self.dropout = nn.Dropout(dropout)
        
        self.pool_type = pool_type
       
        print(f"pool_type {self.pool_type}") 
        if self.pool_type == "attn":
            self.pooler = AttentionPooling(in_dim=self.backbone.config.hidden_size)
        elif self.pool_type == "mean":
            self.pooler = MeanPooling()
        elif self.pool_type == "cls":
            self.pooler = None
        else:
            raise ValueError(f"pool_type {self.pool_type} is not defined")
            
        self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)

        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        if self.pool_type == "attn":
            cls_output, _ = self.pooler(x=hidden_states, attn=hidden_states, mask=attention_mask)
        if self.pool_type == "mean":
            cls_output, _ = self.pooler(x=hidden_states, mask=attention_mask)
        elif self.pool_type == "cls":
            cls_output = hidden_states[:, 0]
        
        x = self.dropout(cls_output)
        logits = self.reg_head(x)
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
