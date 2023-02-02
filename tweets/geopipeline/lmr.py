# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random
import copy

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import prepare_sentences, convert_examples_to_features, read_examples_from_file, get_predictions, write_predictions

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

def get_labels(lmr_mode):
        # The BILOU labels
    if lmr_mode == "TB": # type-less
        labels = ["B-CONT", "B-CTRY", "B-STAT", "B-CNTY", "B-CITY", "B-DIST", "B-NBHD", "B-ISL", "B-NPOI", "B-HPOI", "B-ST", "B-OTHR", 
                "I-CONT", "I-CTRY", "I-STAT", "I-CNTY", "I-CITY", "I-DIST", "I-NBHD", "I-ISL", "I-NPOI", "I-HPOI", "I-ST", "I-OTHR", 
                "L-CONT", "L-CTRY", "L-STAT", "L-CNTY", "L-CITY", "L-DIST", "L-NBHD", "L-ISL", "L-NPOI", "L-HPOI", "L-ST", "L-OTHR", 
                "U-CONT", "U-CTRY", "U-STAT", "U-CNTY", "U-CITY", "U-DIST", "U-NBHD", "U-ISL", "U-NPOI", "U-HPOI", "U-ST", "U-OTHR", 
                "O"]
    else: #"TL": type-less
        labels = ["B-LOC", "I-LOC", "L-LOC", "U-LOC", "O"]
    return labels
    
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

def predict(lines, device, model, tokenizer, glabels, pad_token_label_id):
    torch.cuda.empty_cache()

    dataset, tokens, labels = load_examples(lines, tokenizer, glabels, pad_token_label_id)

    # Note that DistributedSampler samples randomly
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=10)

    preds = None
    out_label_ids = None
    model.eval()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}           
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(glabels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, tokens


def load_examples(sentences, tokenizer, glabels, pad_token_label_id):
    
    #If you need to read BIO-like data from files
    #change lines (list) argument to path (text) argument
    #examples = read_examples_from_file(path, "test")
    # print("in load examples:")
    # print(sentences)
    tokens, labels, examples = prepare_sentences(sentences)
    
    features = convert_examples_to_features(
        examples,
        glabels,
        128,
        tokenizer,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id= 0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, tokens, labels

def get_locations(lines, model, device, lmr_mode):

    #some of the parametres need to be removed
    args = {
        "model_type" : "bert",
        "tokenizer_name": "bert-large-cased",
        "model_name_or_path": "bert-large-cased", 
        "per_gpu_eval_batch_size": 10,
        "max_seq_length": 128, 
        "eval_batch_size": 10,
        "seed": 42,
        "overwrite_cache": True,
        "n_gpu": 0, 
        "no_cuda": True,
        "local_rank": -1
    }

    set_seed(42)

    glabels = get_labels(lmr_mode)
    
    num_labels = len(glabels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    
    tokenizer = BertTokenizer.from_pretrained(args["tokenizer_name"])

    
    predictions, tokens = predict(lines, device, model, tokenizer, glabels, pad_token_label_id)   

    pk, pl, pt = get_predictions(predictions, tokens)

    # print("hereree")
    # print(len(pl))
    
    #In case the size is 0, then append '<nothing>'
    p = []
    for i in range(len(pl)):
        # if(pl[i] == []):
        #     print("herer")
        # print(pl[i])
        p.append(pl[i])
    
    return pk, p
