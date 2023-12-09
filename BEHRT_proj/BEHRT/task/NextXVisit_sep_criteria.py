import os
import sys
sys.path.insert(0, '../')

from torch.utils.data import DataLoader
import pandas as pd
from common.common import create_folder,H5Recorder
import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert

from model import optimiser
import sklearn.metrics as skm
import math
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch
import time
from sklearn.metrics import roc_auc_score
from common.common import load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForSingleLabelPrediction
import warnings
warnings.filterwarnings(action='ignore')

# Provide the path to the parquet file
file_path = "/data/datasets/leyang.sun/merged_age_diagnosis.csv"


# Read the DataFrame from the parquet file
original_data = pd.read_csv(file_path)

original_data['age_vector'] = original_data['age_vector'].apply(lambda x: ''.join([char for char in str(x) if (char != ' ' and char != '[' and char != ']')]).split(','))
original_data['age_vector'] = original_data['age_vector'].apply(lambda x: list(map(int, x)))

original_data['diagnosis_code'] = original_data['diagnosis_code'].apply(lambda x: ''.join([char for char in str(x) if (char != ' ' and char != '[' and char != ']')]).split(','))

# Directory path
directory_path = '/data/datasets/leyang.sun/BEHRT_validation'

# Read CSV files and create a dictionary to store 'deid_pat_ID' from each file
pid_dict = {}
for i in range(1, 7):
    file_path = os.path.join(directory_path, f'file_{i}.csv')
    df = pd.read_csv(file_path)
    pid_dict[f'file_{i}'] = set(df['deid_pat_ID'])


# Assuming original_data is your DataFrame
original_data['label'] = original_data['deid_pat_ID'].apply(lambda x: 1 if x in pid_dict[f'file_{1}'] else 0)

# Save the train and test datasets

file_config = {
    'vocab': '/home/leyang.sun/BERHT/BEHRT/saved_vocab', # token2idx idx2token
    'train': '/home/leyang.sun/BERHT/BEHRT/train_data.parquet',
    'test': '/home/leyang.sun/BERHT/BEHRT/test_data.parquet'
}

def process_patient_data(row):
    # Count the number of visits for the patient
    total_visits = row['diagnosis_code'].count("'SEP'")

    # Check if total visits is greater than 3
    if total_visits <= 3:
        return None

    x_p = row['age_vector']

    label = row['label']

    # Delete elements after the jth 'SEP'
    row['diagnosis_code'] = row['diagnosis_code']

    return pd.Series({'deid_pat_ID': row['deid_pat_ID'], 'age_vector': x_p, 'diagnosis_code': row['diagnosis_code'], 'label': label})

# Apply the function to each row of the original data
processed_data = original_data.apply(process_patient_data, axis=1)

# Drop rows where total visits is less than or equal to 3
processed_data = processed_data.dropna()

# Convert the lists to DataFrames
processed_data_df = pd.DataFrame(processed_data)

processed_data_df['deid_pat_ID'] = processed_data_df['deid_pat_ID'].str.replace('IRB202001139_PAT_', '', regex=False)
processed_data_df['label'] = processed_data_df['label'].astype(str)

from sklearn.model_selection import train_test_split
# Split the data into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(processed_data_df, test_size=0.2, random_state=42)

# Convert the lists to DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


# Rename columns
train_df = train_df.rename(columns={'age_vector': 'age', 'diagnosis_code': 'code', 'deid_pat_ID':'patid'})
test_df = test_df.rename(columns={'age_vector': 'age', 'diagnosis_code': 'code', 'deid_pat_ID':'patid'})


# Reset the index of train and test DataFrames
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Save DataFrames as Parquet files without the index column
train_df.to_parquet('/home/leyang.sun/BERHT/BEHRT/train_data.parquet', index=False)
test_df.to_parquet('/home/leyang.sun/BERHT/BEHRT/test_data.parquet', index=False)




optim_config = {
    'lr': 1e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

global_params = {
    'batch_size': 256,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
    'output_dir': '/home/leyang.sun/BERHT/BEHRT/fine_tuned_model',# output folder
    'best_name': 'FineTuned_BERT_Large_Nextvisit',  # output model name
    'max_len_seq': 100,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5
}
pretrain_model_path = '/home/leyang.sun/BERHT/BEHRT/saved_model/BERT_Large_v1_2023-10-19'

BertVocab = load_obj(file_config['vocab'])

ageVocab, _ = age_vocab(max_age=global_params['max_age'], symbol=global_params['age_symbol'])

labelVocab = {}
label_token = ['0.0','1.0', 'UNK']
for i,x in enumerate(label_token):
    labelVocab[x] = i
print(labelVocab)

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 288, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 12, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
    'num_labels': 1
}

feature_dict = {
    'word':True,
    'seg':True,
    'age':True,
    'position': True
}

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.num_labels = config.get('num_labels')  # Add this line

train = pd.read_parquet(file_config['train'])
Dset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, dataframe=train, max_len=global_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3)

test = pd.read_parquet(file_config['test'])
Dset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, dataframe=test, max_len=global_params['max_len_seq'])
testload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=False, num_workers=3)

# del model
conf = BertConfig(model_config)
model = BertForSingleLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict)


def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()

    # Filter out unnecessary keys and skip the mismatched parameter
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'bert.embeddings.posi_embeddings.weight'
    }

    # Update entries in the existing state dict
    model_dict.update(pretrained_dict)

    # Load the new state dict
    model.load_state_dict(model_dict)
    return model


mode = load_model(pretrain_model_path, model)  # Loading Pretrained Model

model = model.to(global_params['device'])
optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)

# import sklearn
# def precision(logits, label):
#     sig = nn.Sigmoid()
#     output=sig(logits)
#     label, output=label.cpu(), output.detach().cpu()
#     tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
#     return tempprc, output, label

# def precision_test(logits, label):
#     sig = nn.Sigmoid()
#     output=sig(logits)
#     tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy(), average='samples')
#     roc = sklearn.metrics.roc_auc_score(label.numpy(),output.numpy(), average='samples')
#     return tempprc, roc, output, label,

import torch
import torch.nn as nn
import sklearn.metrics


def precision(logits, label):
    sig = nn.Sigmoid()
    output = sig(logits)
    label, output = label.cpu(), output.detach().cpu()

    # Assuming label and output are tensors
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy(), average='macro')  # Change here

    return tempprc, output, label


def precision_test(logits, label):
    sig = nn.Sigmoid()
    output = sig(logits)

    # Assuming label and output are tensors
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy(), average='macro')  # Change here
    roc = sklearn.metrics.roc_auc_score(label.numpy(), output.numpy(), average='macro')  # Change here

    return tempprc, roc, output, label


def train(e):
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0
    for step, batch in enumerate(trainload):
        cnt += 1
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch

        targets = targets.to(torch.long).to(global_params['device'])
        age_ids = age_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        # targets = targets.to(global_params['device'])

        loss, logits = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask, labels=targets)

        if global_params['gradient_accumulation_steps'] > 1:
            loss = loss / global_params['gradient_accumulation_steps']
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if step % 500 == 0:
            prec, a, b = precision(logits, targets)
            print("epoch: {}\t| Cnt: {}\t| Loss: {}\t| precision: {}".format(e, cnt, temp_loss / 500, prec))
            temp_loss = 0

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()


def evaluation():
    model.eval()
    y = []
    y_label = []
    tr_loss = 0
    for step, batch in enumerate(testload):
        model.eval()
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
        targets = torch.argmax(targets, dim=1)

        age_ids = age_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        # targets = targets.to(torch.long).to(global_params['device'])
        targets = targets.to(global_params['device'])

        with torch.no_grad():
            loss, logits = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask, labels=targets)
        logits = logits.cpu()
        targets = targets.cpu()

        tr_loss += loss.item()

        y_label.append(targets)
        y.append(logits)

    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)

    aps, roc, output, label = precision_test(y, y_label)
    return aps, roc, tr_loss

best_pre = 0.0
for e in range(200):
    train(e)
    aps, roc, test_loss = evaluation()
    if aps >best_pre:
        # Save a trained model
        print("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(global_params['output_dir'],global_params['best_name'])
        create_folder(global_params['output_dir'])

        torch.save(model_to_save.state_dict(), output_model_file)
        best_pre = aps
    print('aps : {}'.format(aps))