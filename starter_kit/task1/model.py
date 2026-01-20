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
        self, model_name: str = "bert-base-multilingual-cased", dropout: float = 0.1, pool_type="cls", use_modernbert=False, loss_type="mse", model_class=None
    ) -> None:
        super().__init__()
        if use_modernbert:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            self.backbone = AutoModel.from_pretrained(model_name)
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
        
        self.loss_type = loss_type
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "logcosh":
            pass
        else:
            raise ValueError(f"loss_type {self.loss_type} is not defined")
        

    def forward(self, input_ids, attention_mask, labels=None):
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
        
        loss = None
        if labels is not None:
            if self.loss_type == "mse":
                loss = self.loss_fn(logits, labels)
            elif self.loss_type == "logcosh":
                loss = torch.log(torch.cosh(logits - labels))
                loss = torch.sum(loss)
            else:
                raise ValueError(f"loss_type {self.loss_type} is not defined")

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )

class TransformerVARegressorCls(nn.Module):
    """
    SOTA-style dual-head regressor for Valence/Arousal, based on an encoder-only backbone.

    Differences from baseline TransformerVARegressor:
      - Uses separate regression heads for Valence and Arousal (hidden -> 1 each).
      - Adds separate classification heads for Valence and Arousal (hidden -> n_bins each),
        where the continuous score range [score_min, score_max] is discretized with step = bin_size.
      - If labels are provided, computes a joint loss inside the model:
            loss = lambda_reg * loss_reg + lambda_cls * loss_cls
        where:
            loss_reg = MSE or log-cosh on the regression outputs vs labels;
            loss_cls = 0.5 * (CE(val_logits, v_bin) + CE(aro_logits, a_bin)).
      - The final prediction logits (used by Trainer / evaluation) are an ensemble of
        regression and classification predictions:
            logits = ensemble_alpha * va_reg + (1 - ensemble_alpha) * va_cls.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        dropout: float = 0.1,
        pool_type: str = "cls",
        use_modernbert: bool = False,
        loss_type: str = "mse",
        # SOTA-specific options
        bin_size: float = 0.25,
        use_cls_head: bool = True,
        disable_inner_dropout: bool = True,
        lambda_reg: float = 1.0,
        lambda_cls: float = 1.0,
        ensemble_alpha: float = 0.5,
        score_min: float = 1.0,
        score_max: float = 9.0,
        model_class = None,
        pred_head = "default"
    ) -> None:
        super().__init__()

        # encoder backbone
        if use_modernbert:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout)
        self.pool_type = pool_type
        self.loss_type = loss_type

        print(f"[SOTA] pool_type {self.pool_type}")

        # pooling
        if self.pool_type == "attn":
            self.pooler = AttentionPooling(in_dim=self.backbone.config.hidden_size)
        elif self.pool_type == "mean":
            self.pooler = MeanPooling()
        elif self.pool_type == "cls":
            self.pooler = None
        else:
            raise ValueError(f"pool_type {self.pool_type} is not defined")

        hidden_dim = self.backbone.config.hidden_size

        # regression heads (separate for V and A)
        if pred_head == "default":
            self.val_head = nn.Linear(hidden_dim, 1)
            self.aro_head = nn.Linear(hidden_dim, 1)
        elif pred_head == "normp":
            self.val_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            self.aro_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        else:
            raise ValueError(f"pred_head {pred_head} is not defined.")
            

        # classification heads
        self.use_cls_head = use_cls_head
        self.bin_size = float(bin_size)
        self.lambda_reg = float(lambda_reg)
        self.lambda_cls = float(lambda_cls)
        self.ensemble_alpha = float(ensemble_alpha)
        self.score_min = float(score_min)
        self.score_max = float(score_max)

        if self.use_cls_head:
            # discretize continuous scores into bins
            n_bins = int(round((self.score_max - self.score_min) / self.bin_size)) + 1
            self.n_bins = n_bins
            self.val_cls_head = nn.Linear(hidden_dim, n_bins)
            self.aro_cls_head = nn.Linear(hidden_dim, n_bins)
        else:
            self.n_bins = None
            self.val_cls_head = None
            self.aro_cls_head = None

        if disable_inner_dropout:
            self._disable_encoder_dropout()

        # regression base loss
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "logcosh":
            self.loss_fn = None  # computed manually in forward
        else:
            raise ValueError(f"loss_type {self.loss_type} is not defined")

    # ---------------- internal helpers ----------------

    def _disable_encoder_dropout(self) -> None:
        # change HF config
        if hasattr(self.backbone.config, "hidden_dropout_prob"):
            self.backbone.config.hidden_dropout_prob = 0.0
        if hasattr(self.backbone.config, "attention_probs_dropout_prob"):
            self.backbone.config.attention_probs_dropout_prob = 0.0

        # set all Dropout modules to p=0
        for m in self.backbone.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    def _pool_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pool_type == "attn":
            cls_output, _ = self.pooler(
                x=hidden_states,
                attn=hidden_states,
                mask=attention_mask,
            )
        elif self.pool_type == "mean":
            cls_output, _ = self.pooler(
                x=hidden_states,
                mask=attention_mask,
            )
        elif self.pool_type == "cls":
            cls_output = hidden_states[:, 0]
        else:
            raise ValueError(f"pool_type {self.pool_type} is not defined")

        return cls_output

    def _va_to_bins(
        self,
        va_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_bins is None:
            raise RuntimeError("Classification head is disabled but _va_to_bins was called.")

        v = va_labels[:, 0]
        a = va_labels[:, 1]

        v_idx = torch.clamp(
            ((v - self.score_min) / self.bin_size).long(),
            0,
            self.n_bins - 1,
        )
        a_idx = torch.clamp(
            ((a - self.score_min) / self.bin_size).long(),
            0,
            self.n_bins - 1,
        )
        return v_idx, a_idx

    def _cls_to_scores(
        self,
        val_logits: torch.Tensor,
        aro_logits: torch.Tensor,
    ) -> torch.Tensor:
        val_probs = val_logits.softmax(dim=-1)
        aro_probs = aro_logits.softmax(dim=-1)

        device = val_probs.device
        bin_idx = torch.arange(self.n_bins, device=device).float()
        bin_centers = self.score_min + bin_idx * self.bin_size

        v_cls = (val_probs * bin_centers).sum(dim=-1)
        a_cls = (aro_probs * bin_centers).sum(dim=-1)
        va_cls = torch.stack([v_cls, a_cls], dim=-1)
        return va_cls

    # ---------------- forward ----------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        cls_output = self._pool_hidden(hidden_states, attention_mask)
        x = self.dropout(cls_output)

        # regression outputs
        v_reg = self.val_head(x).squeeze(-1)  # (B,)
        a_reg = self.aro_head(x).squeeze(-1)  # (B,)
        va_reg = torch.stack([v_reg, a_reg], dim=-1)  # (B, 2)

        va_cls = None
        val_logits = None
        aro_logits = None

        if self.use_cls_head and self.val_cls_head is not None:
            val_logits = self.val_cls_head(x)  # (B, n_bins)
            aro_logits = self.aro_cls_head(x)  # (B, n_bins)
            va_cls = self._cls_to_scores(val_logits, aro_logits)

        # ensemble prediction: this is what Trainer / test.py will see as logits
        if va_cls is not None:
            alpha = self.ensemble_alpha
            logits = alpha * va_reg + (1.0 - alpha) * va_cls
        else:
            logits = va_reg

        loss = None
        if labels is not None:
            labels = labels.to(logits.dtype)

            # regression loss
            if self.loss_type == "mse":
                loss_reg = self.loss_fn(va_reg, labels)
            elif self.loss_type == "logcosh":
                diff = va_reg - labels
                loss_reg = torch.log(torch.cosh(diff)).mean()
            else:
                raise ValueError(f"loss_type {self.loss_type} is not defined")

            # classification loss
            if val_logits is not None and aro_logits is not None and self.lambda_cls > 0.0:
                v_idx, a_idx = self._va_to_bins(labels)
                loss_cls_v = nn.functional.cross_entropy(val_logits, v_idx)
                loss_cls_a = nn.functional.cross_entropy(aro_logits, a_idx)
                loss_cls = 0.5 * (loss_cls_v + loss_cls_a)
            else:
                loss_cls = torch.tensor(0.0, device=va_reg.device)

            loss = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )

