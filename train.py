#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    We modified the code based on Alpaca train.py. Author: Zheng Yuan, Hongyi Yuan

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
import numpy as np
import json
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType,
    PeftModel,
)
import math
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch.nn as nn
from copy import deepcopy

# from flashatt import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn() #####in case there is something wrong withflashatten

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    stop_response: bool = field(default=False)
    train_sample_num: int = field(default=2)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lire_weight: float = field(default=100.0)
    length_penalty: float = field(default=1.0)
    only_use_provide: bool = field(default=False)
    only_use_sample: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(ScoreDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            lines = f.readlines()
        self.data = [json.loads(line.strip()) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])


def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    )
    return toked["input_ids"][0]


def stop_response(res):
    stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[: res.find(stop)].strip()
    return res


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool
    num: int

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        score_mask = []
        labels = []
        query_list = []
        for idx, ins in enumerate(instances):
            ins = ins["input_ids"]  # hack
            query = ins["query"]

            # ######test for hh dataset ressponses
            responses = ins["responses"][:self.num]
            scores = ins["scores"][:self.num]
            ###################################

            all_scores.append(scores)
            idxs.append([idx] * len(scores))
            query_list.append(query)

            query_input_ids = _single_tokenize(
                query,
                self.tokenizer,
                max_len=int(self.tokenizer.model_max_length * 2 / 3),
            )
            query_target = torch.LongTensor(
                [IGNORE_INDEX] * (query_input_ids.shape[0] - 1)
            )
            dummy_target = torch.LongTensor([IGNORE_INDEX])
            for i, res in enumerate(responses):
                if self.stop_response:
                    r = stop_response(res)
                else:
                    r = res
                res_input_ids = _single_tokenize(
                    r + self.tokenizer.eos_token,
                    self.tokenizer,
                    max_len=self.tokenizer.model_max_length - query_input_ids.shape[0],
                )  # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(
                    torch.cat((query_target, res_input_ids, dummy_target), dim=0)
                )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
            query=query_list,
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, stop_response=data_args.stop_response,num=data_args.train_sample_num
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


class LIRETrainer(transformers.Trainer):
    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length**self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, rw_scores):
        cand = rw_scores.shape[1]
        new_scores = scores.reshape(-1, cand)  # batch * cand
        diff = new_scores.unsqueeze(1) - new_scores.unsqueeze(-1)  # batch * cand * cand
        rw_diff = rw_scores.unsqueeze(1) - rw_scores.unsqueeze(-1)
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)
        return -diff[aval].sum()

    def lire_loss(self, logits_label, rw_scores):  ###logit_label, rw_scores):
        T = 2.0
        cand = rw_scores.shape[1]
        bz = rw_scores.shape[0]
        logit_label_batch = torch.reshape(
            logits_label, (-1, cand, logits_label.shape[-1])
        )  # batch * cand
        summed_logit = logit_label_batch.sum(-1)
        Q = (summed_logit / T).softmax(dim=-1)
        J = torch.mul(Q, rw_scores.softmax(dim=-1))
        return -J.sum() / bz

    def sft_loss(self, logit_label, rw_scores):
        # max_idx = torch.argmax(rw_scores)
        # return -logit_label[max_idx].mean() #####if batch size=1
        max_idx = torch.argmax(rw_scores, dim=1)  # batch
        cand = rw_scores.shape[1]
        logit_label_batch = torch.reshape(
            logit_label, (-1, cand, logit_label.shape[-1])
        )  # batch * cand * L
        # expert_response_logit_label = torch.gather(logit_label_batch, dim=1, index=max_idx.view(-1, 1, 1).repeat(1, 1, logit_label_batch.size(-1))).squeeze() # batch * L
        expert_response_logit_label = logit_label_batch[
            torch.arange(rw_scores.shape[0]), max_idx
        ].squeeze()
        return -torch.sum(expert_response_logit_label.mean())
    
    def dpo_loss(self,logit_label, logit_label_base, rw_scores):
        cand = rw_scores.shape[1]
        bz = rw_scores.shape[0]
        logit_label_batch = torch.reshape(
            logit_label, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        logit_label_base_batch = torch.reshape(
            logit_label_base, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        summed_logit = logit_label_batch.sum(-1)
        summed_logit_base = logit_label_base_batch.sum(-1)
        policy_chosen_logps = summed_logit[:,0]
        policy_rejected_logps = summed_logit[:,1]
        reference_chosen_logps = summed_logit_base[:,0]
        reference_rejected_logps = summed_logit_base[:,1]
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(0.1 * logits)
        return losses.sum() / bz
    
    def slic_loss(self,logit_label, logit_label_base, rw_scores):
        cand = rw_scores.shape[1]
        bz = rw_scores.shape[0]
        logit_label_batch = torch.reshape(
            logit_label, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        logit_label_base_batch = torch.reshape(
            logit_label_base, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        summed_logit = logit_label_batch.sum(-1)
        summed_logit_base = logit_label_base_batch.sum(-1)
        policy_chosen_logps = summed_logit[:,0]
        policy_rejected_logps = summed_logit[:,1]
        reference_chosen_logps = summed_logit_base[:,0]
        reference_rejected_logps = summed_logit_base[:,1]
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        # logits = pi_logratios - ref_logratios
        losses = torch.clamp(policy_rejected_logps-policy_chosen_logps+1.0,min=0)   ###-reference_chosen_logps

        return losses.sum() / bz

    def load_ref_model(self,model):
        # # add LoRA adaptor
        lora_model = deepcopy(model)
        lora_model.config.use_cache = False

        self.base_model = lora_model

        self.base_model.eval()
        return None

    def stop_response(self, res):
        stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]
        for stop in stops:
            if res.find(stop) >= 0:
                res = res[: res.find(stop)].strip()
        return res

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.only_use_provide:
            inputs["input_ids"] = inputs["input_ids"][-2:]
            inputs["attention_mask"] = inputs["attention_mask"][-2:]
            inputs["labels"] = inputs["labels"][-2:]
            inputs["idxs"] = inputs["idxs"][:, -2:]
            inputs["scores"] = inputs["scores"][:, -2:]
        if self.args.only_use_sample:
            inputs["input_ids"] = inputs["input_ids"][:-2]
            inputs["attention_mask"] = inputs["attention_mask"][:-2]
            inputs["labels"] = inputs["labels"][:-2]
            inputs["idxs"] = inputs["idxs"][:, :-2]
            inputs["scores"] = inputs["scores"][:, :-2]

        logits = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )[
            0
        ]  # (batch * cand) * L * V－－－get output['logits']

        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels"))

        ######add ref model for dpo loss or other loss that require the reference model##################
        # self.base_model = self.base_model.to(model.device)
        # with torch.no_grad():
        #     logits_base = self.base_model(
        #         input_ids=inputs.get("input_ids"),
        #         attention_mask=inputs.get("attention_mask"),
        #     )[0]
        # logits_base_ = F.log_softmax(logits_base, dim=-1)
        # logit_label_base = self.gather_logits_labels(logits_base_, inputs.get("labels"))
        ########################################################################

        # # scores = self.get_score(logit_label, inputs.get("labels"))
        # rrhf_loss = self.rrhf_loss(scores,  inputs.get("scores"))
        lire_loss = self.lire_loss(logit_label, inputs.get("scores"))
        # dpo_loss = self.slic_loss(logit_label,logit_label_base,inputs.get("scores"))
        # sft_loss = self.sft_loss(logit_label, inputs.get("scores"))
        loss = self.args.lire_weight * lire_loss
        return loss
        # return (loss, scores) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # #######to add manually for debugging########
    # import sys

    # sys.argv = "--model_name_or_path

    # parser.add_argument("--model_name_or_path", help="model path")
    # parser.add_argument("--data_path", help="data path")

    ############################################
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )


    # ###### apply lora to llama
    lora_config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        target_modules=["q_proj", "v_proj"],
        r=64,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LIRETrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    ####if DPO loss is applied, initialize the reference model
    # trainer.load_ref_model(model=model)
    ##############################################################
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    train()
