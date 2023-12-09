import os
import sys
sys.path.insert(0, '../')
# %%
from torch.utils.data import DataLoader
import pandas as pd
from common.common import create_folder, H5Recorder
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
from model.NextXVisit import BertForMultiLabelPrediction
import warnings

warnings.filterwarnings(action='ignore')
# %%
# get the list of pid in 6 criteria files create 7 lables
# %%
# create label column: if pid in list =1, not in list =0
# %%
import pandas as pd
import os

# Directory path
directory_path = '/data/datasets/leyang.sun/BEHRT_validation'

# List to store individual DataFrames
dfs = []

# Iterate over CSV files in the directory
for i, file_name in enumerate(os.listdir(directory_path)):
    if file_name.endswith('.csv'):
        # Read CSV into DataFrame
        df = pd.read_csv(os.path.join(directory_path, file_name))

        # Create a label column with the index + 1
        df['label'] = str(i + 1)

        # Append the DataFrame to the list
        dfs.append(df)

# Concatenate DataFrames into a single DataFrame
complete_df = pd.concat(dfs, ignore_index=True)

# %%
# Display the resulting DataFrame
print(complete_df)
# %%
print(complete_df['label'].unique())
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
complete_df['age_vector'] = complete_df['age_vector'].apply(
    lambda x: ''.join([char for char in str(x) if (char != ' ' and char != '[' and char != ']')]).split(','))
complete_df['age_vector'] = complete_df['age_vector'].apply(lambda x: list(map(int, x)))

# Display the result for the first row

# print(original_data['age_vector'][0])
# %%
print(complete_df['age_vector'][0])
# %%

# Assuming 'diagnosis_code' is a column in your DataFrame
complete_df['diagnosis_code'] = complete_df['diagnosis_code'].apply(
    lambda x: ''.join([char for char in str(x) if (char != ' ' and char != '[' and char != ']')]).split(','))
# print(complete_df['diagnosis_code'][0])
# %%
# [ cls, 1,2, sep, 3,4 , sep, 4,5, sep, 2,4, sep,2,sep]
from sklearn.model_selection import train_test_split


def process_patient_data(row):
    # Count the number of visits for the patient
    total_visits = row['diagnosis_code'].count("'SEP'")

    # Check if total visits is greater than 3
    if total_visits <= 3:
        return None

    # Choose a random index j for each patient (3 <= j < total_visits)
    j = np.random.randint(3, total_visits)  # j =4
    # print(j)

    # Create x_p: visits from 1 to j
    x_p = row['age_vector'][:j]

    # Find the (j-1)th and jth 'SEP' indices

    # Assuming diagnosis_code is a list of strings and numbers
    converted_diagnosis_code = []

    sep_indices = [i for i in range(len(row['diagnosis_code'])) if
                   'SEP' in str(row['diagnosis_code'][i])]  # [3,6,9,12,14]

    sep_indices_j_minus_1 = sep_indices[j - 2]  # sep_indices[2] = 9
    sep_indices_j = sep_indices[j - 1]

    label = row['label']

    # Delete elements after the jth 'SEP'
    row['diagnosis_code'] = row['diagnosis_code'][:sep_indices_j]

    return pd.Series(
        {'deid_pat_ID': row['deid_pat_ID'], 'age_vector': x_p, 'diagnosis_code': row['diagnosis_code'], 'label': label})


# Apply the function to each row of the original data
processed_data = complete_df.apply(process_patient_data, axis=1)

# Drop rows where total visits is less than or equal to 3
processed_data = processed_data.dropna()

# Convert the lists to DataFrames
processed_data_df = pd.DataFrame(processed_data)
# %%

# %%

# %%
# print(complete_df.head())
# %%
# Save the train and test datasets

file_config = {
    'vocab': '/home/leyang.sun/BERHT/BEHRT/saved_vocab',  # token2idx idx2token
    'train': '/home/leyang.sun/BERHT/BEHRT/train_data.parquet',
    'test': '/home/leyang.sun/BERHT/BEHRT/test_data.parquet'
}
# %%
processed_data_df['deid_pat_ID'] = processed_data_df['deid_pat_ID'].str.replace('IRB202001139_PAT_', '', regex=False)


# Display the revised DataFrame
# print(complete_df)
# %%
def format_label_vocab(labels):
    labelVocab = {label: idx for idx, label in enumerate(set(labels))}
    return labelVocab


labelVocab = format_label_vocab(processed_data_df['label'])  # Assuming 'label' is the column in your DataFrame
# %%
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(processed_data_df, test_size=0.2, random_state=40)

# Convert the lists to DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
# %%
# Rename columns
train_df = train_df.rename(columns={'age_vector': 'age', 'diagnosis_code': 'code', 'deid_pat_ID': 'patid'})
test_df = test_df.rename(columns={'age_vector': 'age', 'diagnosis_code': 'code', 'deid_pat_ID': 'patid'})

# Reset the index of train and test DataFrames
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Save DataFrames as Parquet files without the index column
train_df.to_parquet('/home/leyang.sun/BERHT/BEHRT/train_data.parquet', index=False)
test_df.to_parquet('/home/leyang.sun/BERHT/BEHRT/test_data.parquet', index=False)

# %%
print(train_df['label'].unique())
# %%
print(test_df['label'].unique())
# %%
optim_config = {
    'lr': 1e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

global_params = {
    'batch_size': 256,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
    'output_dir': '/home/leyang.sun/BERHT/BEHRT/fine_tuned_model',  # output folder
    'best_name': 'FineTuned_BERT_Large_Nextvisit',  # output model name
    'max_len_seq': 100,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5
}
pretrain_model_path = '/home/leyang.sun/BERHT/BEHRT/saved_model/BERT_Large_v1_2023-10-19'

# %%
BertVocab = load_obj(file_config['vocab'])

ageVocab, _ = age_vocab(max_age=global_params['max_age'], symbol=global_params['age_symbol'])
# %%
model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()),  # number of disease + symbols for word embedding
    'hidden_size': 288,  # word embedding and seg embedding hidden size
    'seg_vocab_size': 2,  # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()),  # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'],  # maximum number of tokens
    'hidden_dropout_prob': 0.1,  # dropout rate
    'num_hidden_layers': 6,  # number of multi-head attention layers required
    'num_attention_heads': 12,  # number of attention heads
    'attention_probs_dropout_prob': 0.1,  # multi-head attention dropout rate
    'intermediate_size': 512,  # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu',
    # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02,  # parameter weight initializer range
}

feature_dict = {
    'word': True,
    'seg': True,
    'age': True,
    'position': True
}


# %%
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
            max_position_embeddings=config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')


# %%
train = pd.read_parquet(file_config['train'])
Dset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, dataframe=train,
                 max_len=global_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=True, num_workers=3, drop_last=True)
# %%
test = pd.read_parquet(file_config['test'])
Dset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, dataframe=test,
                 max_len=global_params['max_len_seq'])
testload = DataLoader(dataset=Dset, batch_size=global_params['batch_size'], shuffle=False, num_workers=3)
# %%
# del model
conf = BertConfig(model_config)
model = BertForMultiLabelPrediction(conf, num_labels=6, feature_dict=feature_dict)


# %%
# def load_model(path, model):
#     # load pretrained model and update weights
#     pretrained_dict = torch.load(path)
#     model_dict = model.state_dict()
#     # 1. filter out unnecessary keys
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict)
#     # 3. load the new state dict
#     model.load_state_dict(model_dict)
#     return model

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
# %%
model = model.to(global_params['device'])
optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)
# %%
import sklearn


def precision(logits, label):
    sig = nn.Sigmoid()
    output = sig(logits)
    label, output = label.cpu(), output.detach().cpu()
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy(), average='samples')
    return tempprc, output, label


def precision_test(logits, label):
    sig = nn.Sigmoid()
    output = sig(logits)
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy(), average='samples')
    roc = sklearn.metrics.roc_auc_score(label.numpy(), output.numpy(), average='samples')
    return tempprc, roc, output, label,


# %%
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer(classes=list(labelVocab.values()))
# mlb.fit([[each] for each in list(labelVocab.values())])
# %%
def train(e, train_losses):
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0
    for step, batch in enumerate(trainload):
        cnt += 1
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch

        # targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        age_ids = age_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
        targets = targets.to(global_params['device'])

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

    train_losses.append(tr_loss / nb_tr_steps)


def evaluation():
    model.eval()
    y = []
    y_label = []
    tr_loss = 0
    for step, batch in enumerate(testload):
        model.eval()
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)

        age_ids = age_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params['device'])
        posi_ids = posi_ids.to(global_params['device'])
        segment_ids = segment_ids.to(global_params['device'])
        attMask = attMask.to(global_params['device'])
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


# %%
best_pre = 0.0
train_losses = []
for e in range(100):
    train(e, train_losses)
    aps, roc, test_loss = evaluation()
    if aps > best_pre:
        # Save a trained model
        print("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(global_params['output_dir'], global_params['best_name'])
        create_folder(global_params['output_dir'])

        torch.save(model_to_save.state_dict(), output_model_file)
        best_pre = aps
    print('aps : {}'.format(aps))