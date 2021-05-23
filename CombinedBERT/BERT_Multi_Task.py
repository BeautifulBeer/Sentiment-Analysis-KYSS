#!/usr/bin/env python
# coding: utf-8

# # BERT Sentiment Classification
# ## Applied Methodology
# - Multi-task learning
# 
# ## Available choices
# - Pretrained model : RoBERTa, BERT
# - Tasks : Kaggle, IMDB, Sarcasm

# In[ ]:


get_ipython().system('pip3 install torch')
get_ipython().system('pip3 install torchtext')
get_ipython().system('pip3 install transformers')
get_ipython().system('pip3 install tqdm')
get_ipython().system('pip3 install pathlib')


# In[ ]:


import re
import torch
import torch.nn as nn
import torchtext
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from enum import Enum
from pprint import pprint
from torch.utils.data import DataLoader, Dataset

# Label for Task
SENTIMENT_LABEL = 'sentiment'
IMDB_LABEL = 'imdb'
SARCASM_LABEL = 'sarcasm'

# Label for loss function
CROSS_ENTROPY_LOSS_LABEL = 'crossentropyloss'

# Configurations for running code
configs = {
    'path' : {
        'colab' : {
            'root' : '/content/gdrive/My Drive',
            'data' : 'dataset',
            'checkpoint' : 'checkpoints',
        },
        'local' : {
            'root' : '../',
            'data' : '.data',
            'checkpoint' : 'checkpoints',
        },   
    },
    'task' : {
        SENTIMENT_LABEL : {
            'train_file' : 'sentence-classification/train_final.csv',
            'test_file' : 'sentence-classification/eval_final_open.csv',
            'train_valid_frac' : 0.8,
            'out_features' : 5,
            'loss_fn' : CROSS_ENTROPY_LOSS_LABEL,
        },
        IMDB_LABEL : {
            'train_file' : 'imdb/train_IM.csv',
            'train_valid_frac' : 0.8,
            'out_features' : 2,
            'loss_fn' : CROSS_ENTROPY_LOSS_LABEL,
        },
        SARCASM_LABEL : {
            'train_file' : 'reddit-sarcasm/train_sarcasm.csv',
            'train_valid_frac' : 0.8,
            'out_features' : 2,
            'loss_fn' : CROSS_ENTROPY_LOSS_LABEL,
        },
    }
}

# Value for each Task
class Task(Enum):
    SENTIMENT = 1
    IMDB = 2
    SARCASM = 3

# To store trained model's checkpoint(filename)
start_time = datetime.now()

if 'google.colab' in str(get_ipython()):
    print('Running on CoLab')
    from google.colab import drive

    drive.mount('/content/gdrive')

    root_dir = Path(configs['path']['colab']['root'])
    data_dir = Path(root_dir, configs['path']['colab']['data'])
    checkpoint_dir = Path(root_dir, configs['path']['colab']['checkpoint'])

else:
    print('Not running on CoLab')
    
    root_dir = Path(configs['path']['local']['root'])
    data_dir = Path(root_dir, configs['path']['local']['data'])
    checkpoint_dir = Path(root_dir, configs['path']['local']['checkpoint'])

Path(checkpoint_dir).mkdir(parents=False, exist_ok=True)

# Load pretrained model
# PRETRAINED_MODEL = 'bert-base-cased'
PRETRAINED_MODEL = 'roberta-base'

if re.compile('^robert').match(PRETRAINED_MODEL):
    from transformers import RobertaModel, RobertaTokenizer, AdamW
    
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)
    bert_model = RobertaModel.from_pretrained(PRETRAINED_MODEL)

elif re.compile('^bert').match(PRETRAINED_MODEL):
    from transformers import BertTokenizer, BertModel, AdamW

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    bert_model = BertModel.from_pretrained(PRETRAINED_MODEL)


torch.cuda.empty_cache()


# ## Preprocessing Dataset

# In[ ]:


class ClassificationDataset(Dataset):
    '''
    Preprocessing dataframe to dataset. CSV should have two columns : "Sentence", "Category"
    :params
      df: dataframe loaded from csv
      task: value for each task - reference class Task(Enum)
    '''
    def __init__(self, df, task, tokenizer, max_len, is_train):
        self.task = task
        self.sentences = df['Sentence'].to_numpy()
        self.is_train = is_train
        if self.is_train:
            self.targets = df['Category'].to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        if self.is_train:
            target = self.targets[idx]
               
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens = True, # Add CLS, SEP
            max_length = self.max_len,
            return_token_type_ids = False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        if self.is_train:
            return {
                'task' : self.task,
                'text' : sentence,
                'input_ids' : encoding['input_ids'].flatten(),
                'attention_mask' : encoding['attention_mask'].flatten(),
                'targets' : torch.tensor(target, dtype=torch.long)
            }
        else:
            return {
                'task' :self.task,
                'text' : sentence,
                'input_ids' : encoding['input_ids'].flatten(),
                'attention_mask' : encoding['attention_mask'].flatten(),
            }

def load_csv_data(configs, seed):
    '''
    Return dictionary of dataframes for each task from csv
    :return
      dict, dict, dict: train, valid, test dataset is returned. Task label is key (e.g. SENTIMENT_LABEL)
    '''
    train_data = {}
    valid_data = {}
    test_data = {}

    for task in configs['task']:
        raw_data = pd.read_csv(data_dir.joinpath(configs['task'][task]['train_file']))
        if task == 'imdb':
            raw_data.rename(columns = {'Phrase': 'Sentence', 'Sentiment': 'Category'}, inplace=True)
        train_data[task] = raw_data.sample(frac=configs['task'][task]['train_valid_frac'], random_state=seed)
        valid_data[task] = raw_data.drop(train_data[task].index)
        if 'test_file' in configs['task'][task]:
            test_data[task] = pd.read_csv(data_dir.joinpath(configs['task'][task]['test_file']))
            if task == 'imdb':
                test_data[task].rename(columns = {'Phrase': 'Sentence', 'Sentiment': 'Category'}, inplace=True)
    
    return train_data, valid_data, test_data

def print_dataset_configs(configs, train, valid, test):
    '''
    Print overview of preprocessed dataset
    '''
    for task in train:
        print(f'{task} dataset')
        print(f'='*25)
        print(f'Train/Valid : {configs["task"][task]["train_valid_frac"]:.2f}/{1-configs["task"][task]["train_valid_frac"]:.2f}')
        print(f'='*25)
        print(f'Train dataset length : {len(train[task])}')
        if task in valid:
            print(f'Valid dataset length : {len(valid[task])}')
        if task in test:
            print(f'Test dataset length : {len(test[task])}')
        print(f'='*25)
        print('')

def get_data_loader(phase, task_df, tokenizer, max_len, batch_size, is_train, shuffle):
    '''
    Get an entire dataloader. Each dataset of a task is preprocessed under the same conditions (e.g. batch_size)
    :params
      task_df: dataset for each task, dataframe
    '''
    total_dataset = []
    
    for task in task_df:
        dataset = ClassificationDataset(
            task_df[task],
            convert_label_to_enum(task),
            tokenizer = tokenizer,
            max_len = max_len,
            is_train=is_train,
        )
        time.sleep(1)
        
        loader = DataLoader(
            dataset,
            batch_size = batch_size
        )
        
        print(f'Combine {phase} - {task} dataset')
        time.sleep(1)
        for batch in tqdm(loader):
            total_dataset.append(batch)
    
    return DataLoader(
        total_dataset,
        shuffle = shuffle,
        batch_size = 1,
    )

def convert_enum_to_label(enum):
    if enum == Task.SENTIMENT.value:
        return SENTIMENT_LABEL
    elif enum == Task.IMDB.value:
        return IMDB_LABEL
    elif enum == Task.SARCASM.value:
        return SARCASM_LABEL

def convert_label_to_enum(label):
    if label == SENTIMENT_LABEL:
        return Task.SENTIMENT.value
    elif label == IMDB_LABEL:
        return Task.IMDB.value
    elif label == SARCASM_LABEL:
        return Task.SARCASM.value
    
def convert_name_to_func(name):
    if name == CROSS_ENTROPY_LOSS_LABEL:
        return nn.CrossEntropyLoss()
    elif name == BCE_LOSS_WITH_LOGITS_LABEL:
        return nn.BCEWithLogitsLoss()


# ## Define model, Train, Valid, Prediction

# In[ ]:


class SentimentModel(nn.Module):
    '''
    Multi-task learning is applied
    fc_sent, fc_im, fc_sarc is fully connected layer of each task and train separately. (share BERT layer)
    '''
    def __init__(self, bert, configs, dropout_p):
        super(SentimentModel, self).__init__()
        self.bert = bert
        self.dropout_p = dropout_p
        hidden_size = bert.config.to_dict()['hidden_size']
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        if SENTIMENT_LABEL in configs['task']:
            self.fc_sent = nn.Linear(
                hidden_size,
                configs['task'][SENTIMENT_LABEL]['out_features'],
            )
        
        if IMDB_LABEL in configs['task']:
            self.fc_im = nn.Linear(
                hidden_size,
                configs['task'][IMDB_LABEL]['out_features'],
            )

        if SARCASM_LABEL in configs['task']:
            self.fc_sarc = nn.Linear(
                hidden_size,
                configs['task'][SARCASM_LABEL]['out_features'],
            )
        
    def forward(self, input_ids, attention_mask, target_task):
        '''
          forward for each task
          :params
            target_task: task label(string)
        '''
        result = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        out = self.dropout(result.pooler_output)

        if target_task == SENTIMENT_LABEL:
            out = self.fc_sent(out)
        elif target_task == IMDB_LABEL:
            out = self.fc_im(out)
        elif target_task == SARCASM_LABEL:
            out = self.fc_sarc(out)
        
        if not self.train:
            out = out * (1-self.dropout_p)

        return out

def train_epoch(model, loader, loss_fn, optimizer, scheduler, dataset_size):    
    losses = {}
    means = {}
    correct_predictions = {}
    
    for task in loss_fn:
        losses[task] = []
        correct_predictions[task] = 0.0
    
    model = model.train()
    
    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]
        targets = batch['targets'][0]
        task = convert_enum_to_label(batch['task'][0][0])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = targets.cuda()
        
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            target_task = task,
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn[task](outputs, targets)
        
        correct_predictions[task] += torch.sum(preds == targets)
        losses[task].append(loss.detach().item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    for task in correct_predictions:
        correct_predictions[task] = correct_predictions[task].double() / dataset_size[task]
        means[task] = np.sum(losses[task]) / dataset_size[task]
        
    return correct_predictions, losses, means

def valid_epoch(model, loader, loss_fn, dataset_size):
    losses = {}
    means = {}
    correct_predictions = {}

    for task in loss_fn:
        losses[task] = []
        correct_predictions[task] = 0.0
    
    model = model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'][0]
            attention_mask = batch['attention_mask'][0]
            targets = batch['targets'][0]
            task = convert_enum_to_label(batch['task'][0][0])
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                targets = targets.cuda()
                
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                target_task = task
            )
            
            _, preds = torch.max(outputs, dim=1)
            
            loss = loss_fn[task](outputs, targets)
            
            correct_predictions[task] += torch.sum(preds == targets)
            losses[task].append(loss.detach().item())

        for task in correct_predictions:
            correct_predictions[task] = correct_predictions[task].double() / dataset_size[task]
            means[task] = np.sum(losses[task]) / dataset_size[task] 

    return correct_predictions, losses, means

def get_predictions(model, loader, task):
    model = model.eval()
    
    predictions = []
    predictions_probs = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'][0]
            attention_mask = batch['attention_mask'][0]
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                target_task = task
            )         
            predictions.extend(torch.argmax(outputs, dim=1))
            
    return torch.stack(predictions).cpu()


def print_model_results(phase, epoch, accuracy, losses):
    for task in accuracy:
        print(f'{phase} : {task} accruacy/loss : {accuracy[task]:.5f}/{losses[task]}')


# ### Load Dataset

# In[ ]:


RANDOM_SEED = 884532
# For same result
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

max_len = 100
batch_size = 8

train_data, valid_data, test_data = load_csv_data(configs, RANDOM_SEED)


print(f'Batch size = {batch_size}')
print(f'-'*50)
print(f'Task Configuration')
print(f'-'*50)
print_dataset_configs(configs, train_data, valid_data, test_data)
print(f'-'*50)

train_loader = get_data_loader('Train', train_data, tokenizer, max_len, batch_size, True, True)
valid_loader = get_data_loader('Valid', valid_data, tokenizer, max_len, batch_size, True, True)
test_loader = get_data_loader('Test', test_data, tokenizer, max_len, batch_size, False, False)


# ### Train, Valid, Test model

# In[ ]:


model = SentimentModel(bert_model, configs, 0.1)

epochs = 10
total_steps = len(train_loader) * epochs
learning_rate = 2e-5

loss_fn = {}
train_size = {}
valid_size = {}

for task in configs['task']:
    loss_fn[task] = convert_name_to_func(configs['task'][task]['loss_fn'])
    train_size[task] = len(train_data[task])
    valid_size[task] = len(valid_data[task])

if torch.cuda.is_available():
    model = model.cuda()
    for task in loss_fn:
        # model fully connected layer cuda
        loss_fn[task] = loss_fn[task].cuda()
    
# Adam optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Cosine annealing warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0 = int(epochs / 3),
    T_mult = 1,
    eta_min = 2e-8
)

results = {
    'train_loss' : [],
    'train_acc' : [],
    'valid_loss' : [],
    'valid_acc' : []
}

best_valid_acc = 0

for epoch in range(epochs):
    print(f'Epoch {epoch + 1} / {epochs}')
    time.sleep(1)
    train_acc, train_loss, train_loss_mean = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        scheduler,
        train_size
    )
    print_model_results('Train', epoch, train_acc, train_loss_mean)
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    time.sleep(1)
    valid_acc, valid_loss, valid_loss_mean = valid_epoch(
        model,
        valid_loader,
        loss_fn,
        valid_size
    ) 
    print_model_results('Valid', epoch, valid_acc, valid_loss_mean)
    results['valid_loss'].append(valid_loss)
    results['valid_acc'].append(valid_acc)
    
    if best_valid_acc < valid_acc[SENTIMENT_LABEL]:
        best_valid_acc = valid_acc[SENTIMENT_LABEL]
        torch.save(model.state_dict(), Path(checkpoint_dir, f'Model_Valid_{start_time.strftime("%Y_%m_%d_%H_%M_%S")}.pt'))
        print(f'Best valid acc : {best_valid_acc * 100:.5f}%')
        
    print(f'-'*25)


torch.save(model.state_dict(), Path(checkpoint_dir, f'Model_Train_{start_time.strftime("%Y_%m_%d_%H_%M_%S")}.pt'))
model.load_state_dict(torch.load(Path(checkpoint_dir, f'Model_Valid_{start_time.strftime("%Y_%m_%d_%H_%M_%S")}.pt')))
predictions = get_predictions(model, test_loader, SENTIMENT_LABEL)

submission = pd.DataFrame({'Id' : range(len(predictions)), 'Category' : predictions})
submission.to_csv('submission.csv', index=False)

