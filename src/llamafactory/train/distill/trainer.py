from copy import deepcopy
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override
from trl.models import PreTrainedModelWrapper

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than

from ..sft.trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, PreTrainedModel
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)


def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


class CustomDistillTrainer(CustomSeq2SeqTrainer):
    def __init__(
        self,
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(finetuning_args, processor, gen_kwargs, **kwargs)
        
        # teacher model for distillation
        self.ref_model = ref_model 
        self.temperature = 1.

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
                    self.ref_model.eval()
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        return model
    
    @override
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs=False, 
        num_items_in_batch=None, 
        *args, 
        **kwargs,
    ):
        # forward distillation KL loss
        inputs = {**inputs, **kwargs}
        outputs = model(**inputs, output_hidden_states=True)
        stu_logits = outputs["logits"]

        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs, output_hidden_states=True)
            ref_logits = ref_outputs["logits"]

        # if vocab_size are different
        stu_logits, ref_logits = pad_logits(stu_logits, ref_logits)

        ref_probs = F.softmax(ref_logits / self.temperature, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(stu_logits)
        stu_logprobs = F.log_softmax(stu_logits / self.temperature, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(ref_probs * stu_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        loss_mask = (inputs["labels"] != IGNORE_INDEX).int()
        distill_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
        distill_loss = distill_loss * (self.temperature ** 2)

        lm_loss = outputs["loss"]
        loss = distill_loss + lm_loss


        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
