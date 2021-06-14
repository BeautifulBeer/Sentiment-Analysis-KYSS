# Importing the libraries needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW
import logging
from pathlib import Path
import time
logging.basicConfig(level=logging.ERROR)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

root_dir = Path('../')
data_dir = Path(root_dir,'data', 'sentence-classification')
print("Using {} device".format(device))

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, is_train):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Sentence
        self.is_train = is_train
        if self.is_train:
            self.targets = self.data.Category
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        if self.is_train:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            }

def get_data_loader(df, tokenizer, max_len, batch_size, is_train, shuffle):
    dataset = SentimentData(
        df,
        tokenizer = tokenizer,
        max_len = max_len,
        is_train=is_train,
    )

    return DataLoader(
        dataset,
        shuffle = shuffle,
        batch_size=batch_size
    )

def get_predictions(model, loader):
    model = model.eval()

    predictions = []
    predictions_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['ids']
            attention_mask = batch['mask']
            token_type_ids = batch["token_type_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )

            _, preds = torch.max(outputs, dim=1)

            predictions.extend(torch.argmax(outputs, dim=1))
    return torch.stack(predictions).cpu()

# Defining some key variables that will be used later on in the training
load_model = True # EDIT
load_model_path = "domain_roberta_sentiment_model_50.pt" # EDIT
load_optimizer_path = "domain_roberta_sentiment_optimizer_50.pt" # EDIT
output_model_name = 'domain_roberta_sentiment_model' # EDIT
output_optimizer_name = 'domain_roberta_sentiment_optimizer' # EDIT
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)
max_len = 128 # 256
batch_size = 64*2 # 64
LEARNING_RATE = 2e-05
train_valid_frac = 0.8

# 원래 데이터셋
train = pd.read_csv(data_dir.joinpath('merged_250742_train_plus.csv')) # EDIT
new_df = train[['Sentence', 'Category']]

train_df=new_df.sample(frac=train_valid_frac,random_state=200)
valid_df=new_df.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

test_df = pd.read_csv(data_dir.joinpath('eval_final_open.csv'))
test_df = test_df[['Sentence']]

print(f'Dataset Configuration')
print(f'-'*25)
print(f'Train/Valid = {train_valid_frac:.2f}/{1-train_valid_frac:.2f}')
print(f'Batch size = {batch_size}')
print(f'-'*25)
print(f'Train set : {len(train_df)}')
print(f'Valid set : {len(valid_df)}')
print(f'Test set : {len(test_df)}')

training_loader   = get_data_loader(train_df, tokenizer, max_len, batch_size, True, True)
validating_loader = get_data_loader(valid_df, tokenizer, max_len, batch_size, True, True)
testing_loader    = get_data_loader(test_df, tokenizer, max_len, batch_size, False, False)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model = torch.nn.DataParallel(model)
if load_model:
    model.load_state_dict(torch.load(load_model_path))
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
if load_model:
    optimizer.load_state_dict(torch.load(load_optimizer_path))

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# Defining the training function on the 80% of the dataset for tuning the distilbert model
def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for data in tqdm(training_loader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

def valid(model):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for data in tqdm(validating_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()

            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu

best_acc = 0
START_EPOCH = 50 # EDIT
EPOCHS = 80 # EDIT
for epoch in range(START_EPOCH, EPOCHS):
    time.sleep(1)
    train(epoch)

    time.sleep(1)
    acc = valid(model)

    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{output_model_name}_{epoch+1}_best.pt")
        torch.save(optimizer.state_dict(), f"{output_optimizer_name}_{epoch+1}_best.pt")
        continue

    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"{output_model_name}_{epoch+1}.pt")
        torch.save(optimizer.state_dict(), f"{output_optimizer_name}_{epoch+1}.pt")

predictions = get_predictions(model, testing_loader)
submission = pd.DataFrame({'Id' : range(len(predictions)), 'Category' : predictions})
submission.to_csv('domain_submission_80.csv', index=False) # EDIT

print('All files saved')
print('This tutorial is completed')
