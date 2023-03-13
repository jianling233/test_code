import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
import shutil

from torch.utils.data import DataLoader, Dataset
# import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
#import tokenizers
#import transformers
#print(f"tokenizers.__version__: {tokenizers.__version__}")
#print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
# %env TOKENIZERS_PARALLELISM=true

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CFG:
    wandb = False
    apex = True
    model = 'microsoft/deberta-v3-large'
    seed = 42
    max_len = 512
    dropout = 0.2
    target_size=3
    n_accumulate=1
    print_freq = 100
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    scheduler = 'cosine'
    batch_size = 4
    num_workers = 8
    lr = 2e-5
    weight_decay = 0.01
    epochs = 1
    n_fold = 5
    trn_fold=[i for i in range(n_fold)]
    train = True 
    num_warmup_steps = 0
    num_cycles=0.5
    encoder_lr=2e-5
    decoder_lr=2e-5
    debug = True
    debug_ver2 = False
    gradient_checkpoint=False
    
OUTPUT_DIR = './baseline/'
model_path = '../input/deberta-v3-large'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

# def criterion(outputs, labels):
#     return nn.BCEWithLogitsLoss(reduction="mean")(outputs, labels)

def get_score(outputs, labels):
    outputs = F.softmax(torch.tensor(outputs)).numpy()
    score = log_loss(labels,outputs)
    return round(score, 5)


INPUT_DIR = "../input/feedback-prize-effectiveness/"
train = pd.read_csv(INPUT_DIR+'train.csv')
test = pd.read_csv(INPUT_DIR+'test.csv')
print(train.head())
print(train.shape)
print(test.head())
print(test.shape)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

train['essay_text']  = train['essay_id'].apply(lambda x: get_essay(x, is_train=True))

train['discourse_text'] = train['discourse_text'].apply(lambda x : resolve_encodings_and_normalize(x))
train['essay_text'] = train['essay_text'].apply(lambda x : resolve_encodings_and_normalize(x))


train['essay_text']  = train['essay_id'].apply(lambda x: get_essay(x, is_train=True))
SEP = tokenizer.sep_token
train['text'] = train['discourse_type'] + ' ' + train['discourse_text'] + SEP + train['essay_text']
print(train.head())

gkf = GroupKFold(n_splits=CFG.n_fold)

for fold, ( _, val_) in enumerate(gkf.split(X=train, groups=train.essay_id)):
    train.loc[val_ , "fold"] = int(fold)
    
train["fold"] = train["fold"].astype(int)
train.groupby('fold')['discourse_effectiveness'].value_counts()

train['label'] = train['discourse_effectiveness'].map({'Ineffective':0, 'Adequate':1, 'Effective':2})

# skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
# train['fold'] = -1
# train['label'] = train['discourse_effectiveness'].map({'Ineffective':0, 'Adequate':1, 'Effective':2})
# for i, (_, val_) in enumerate(skf.split(train, train['label'])):
#     train.loc[val_, 'fold'] = int(i)
# train.fold.value_counts()

class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = CFG.max_len
        self.text = df['text'].values
        self.tokenizer = CFG.tokenizer
        self.targets = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length = self.max_len
        )
        return {
            'input_ids':inputs['input_ids'],
            'attention_mask':inputs['attention_mask'],
            'target':self.targets[index]
            }
    
class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = Collate(CFG.tokenizer, isTrain=True)


# class MultiSampleDropout(nn.Module):
#     def __init__(self, max_dropout_rate, num_samples, classifier):
#         super(MultiSampleDropout, self).__init__()
#         self.dropout = nn.Dropout       
#         self.classifier = classifier
#         self.max_dropout_rate = max_dropout_rate
#         self.num_samples = num_samples
        
#     def forward(self, out):
#         return torch.mean(torch.stack([
#             self.classifier(self.dropout(p=self.max_dropout_rate)(out))
#             for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))
#         ], dim=0), dim=0)

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 5, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
    
class FeedBackModel(nn.Module):
    def __init__(self, model_name,layer_start):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        config = self.config
        config.update({
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
        })
        self.model = AutoModel.from_pretrained(model_name,config=config)
        self.layer_start = layer_start
        self.pooling = WeightedLayerPooling(config.num_hidden_layers,
                                            layer_start=layer_start,
                                            layer_weights=None)
        self.mean_pooler = MeanPooling()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.drop = nn.Dropout(p=0.1)
        self.fc = nn.Linear(config.hidden_size, 3)
        # self.multisample_dropout = MultiSampleDropout(0.2, num_samples=8, classifier=self.fc)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask)
        # all_hidden_states = torch.stack(out.hidden_states)
        # weighted_pooling_embeddings = self.layer_norm(self.pooling(all_hidden_states))
        # weighted_pooling_embeddings = weighted_pooling_embeddings[:,0]
        # norm_embeddings = self.drop(weighted_pooling_embeddings)
        # outputs = self.fc(norm_embeddings)
        out = self.layer_norm(self.pooling(torch.stack(out.hidden_states)))
        out = self.mean_pooler(out, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs

# class MeanPooling(nn.Module):
#     def __init__(self):
#         super(MeanPooling, self).__init__()
        
#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings

# class MultiSampleDropout(nn.Module):
#     def __init__(self, max_dropout_rate, num_samples, classifier):
#         super(MultiSampleDropout, self).__init__()
#         self.dropout = nn.Dropout       
#         self.classifier = classifier
#         self.max_dropout_rate = max_dropout_rate
#         self.num_samples = num_samples
        
#     def forward(self, out):
#         return torch.mean(torch.stack([
#             self.classifier(self.dropout(p=self.max_dropout_rate)(out))
#             for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))
#         ], dim=0), dim=0)
    
# class FeedBackModel(nn.Module):
#     def __init__(self, model_name):
#         super(FeedBackModel, self).__init__()
#         self.model = AutoModel.from_pretrained(model_name)
#         self.config = AutoConfig.from_pretrained(model_name)
#         self.pooler = MeanPooling()
#         self.fc = nn.Linear(self.config.hidden_size,3)
#         self.multisample_dropout = MultiSampleDropout(0.2, num_samples=8, classifier=self.fc)
        
#     def forward(self, ids, mask):        
#         out = self.model(input_ids=ids,attention_mask=mask,
#                          output_hidden_states=False)
#         out = self.pooler(out.last_hidden_state, mask)
#         outputs = self.multisample_dropout(out)
        
#         return outputs


    
def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0
    start = end = time.time()

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)
        
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)

        #accumulate
        loss = loss / CFG.n_accumulate 
        loss.backward()
        if (step +1) % CFG.n_accumulate == 0:
            optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  .format(epoch+1, step, len(dataloader), 
                          remain=timeSince(start, float(step+1)/len(dataloader))))

    gc.collect()

    return epoch_loss
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0

    start = end = time.time()
    pred = []

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)
        pred.append(outputs.to('cpu').numpy())

        running_loss += (loss.item()* batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  .format(step, len(dataloader),
                          remain=timeSince(start, float(step+1)/len(dataloader))))
            
    pred = np.concatenate(pred)
            
    return epoch_loss, pred

def train_loop(fold):
    #wandb.watch(model, log_freq=100)

    LOGGER.info(f'-------------fold:{fold} training-------------')

    train_data = train[train.fold != fold].reset_index(drop=True)
    valid_data = train[train.fold == fold].reset_index(drop=True)
    valid_labels = valid_data.label.values

    trainDataset = FeedBackDataset(train_data, CFG.tokenizer, CFG.max_len)
    validDataset = FeedBackDataset(valid_data, CFG.tokenizer, CFG.max_len)

    train_loader = DataLoader(trainDataset,
                              batch_size = CFG.batch_size,
                              shuffle=True,
                              collate_fn = collate_fn,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=True)
    
    valid_loader = DataLoader(validDataset,
                              batch_size = CFG.batch_size,
                              shuffle=False,
                              collate_fn = collate_fn,
                              num_workers = CFG.num_workers,
                              pin_memory = True,
                              drop_last=False)
    model = FeedBackModel(model_path,layer_start=14)
    # model = CustomModel(model_path,config_path=None, pretrained=True)
    # model = FeedBackModel(model_path)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.lr, eps=CFG.eps, betas=CFG.betas)
    num_train_steps = int(len(train_data) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    best_score = 100

    for epoch in range(CFG.epochs):
        start_time = time.time()

        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch)
        valid_epoch_loss, pred = valid_one_epoch(model, valid_loader, device, epoch)

        score = get_score(pred, valid_labels)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": train_epoch_loss, 
                       f"[fold{fold}] avg_val_loss": valid_epoch_loss,
                       f"[fold{fold}] score": score})
            
        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': pred},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            
    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_data['pred_0'] = predictions[:, 0]
    valid_data['pred_1'] = predictions[:, 1]
    valid_data['pred_2'] = predictions[:, 2]


    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_data

if __name__ == '__main__':
    
    def get_result(oof_df):
        labels = oof_df['label'].values
        preds = oof_df[['pred_0', 'pred_1', 'pred_2']].values.tolist()
        score = get_score(preds, labels)
        LOGGER.info(f'Score: {score:<.4f}')
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')
        oof_df.to_csv(OUTPUT_DIR+f'oof_df.csv', index=False)