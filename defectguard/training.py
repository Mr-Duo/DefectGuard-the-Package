import os, torch, pickle
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import torch.nn as nn
from .models import (
    DeepJIT,
    CC2Vec,
    SimCom,
    LAPredict,
    LogisticRegression,
    TLELModel as TLEL,
    JITLine,
)
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from .utils.padding import padding_data
from .utils.utils import open_jsonl
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    return lr_auc

def init_model(model_name, language, device):   
    if  model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif  model_name == "cc2vec":
        return CC2Vec(language=language, device=device)
    elif  model_name == "simcom":
        return SimCom(language=language, device=device)
    elif  model_name == "lapredict":
        return LAPredict(language=language)
    elif  model_name == "tlel":
        return TLEL(language=language)
    elif  model_name == "jitline":
        return JITLine(language=language)
    elif  model_name == "lr":
        return LogisticRegression(language=language)
    else:
        raise Exception("No such model")

class CustomDataset(Dataset):
    def __init__(self, data, code_dict, msg_dict, hyperparameters):
        self.data = data
        self.code_dict = code_dict
        self.msg_dict = msg_dict
        self.hyperparameters = hyperparameters
        
        self.id = [item["commit_id"] for item in self.data]
        self.codes = [item["code_change"] for item in self.data]
        self.messages = [item["messages"] for item in self.data]
        self.labels = [item["label"] for item in self.data]
        self.codes = padding_data(data=self.codes, dictionary=self.code_dict, params=self.hyperparameters, type='code')
        self.messages = padding_data(data=self.messages, dictionary=self.msg_dict, params=self.hyperparameters, type='msg')
        
        self.data = None
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        commit_hash = self.id[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        code = torch.tensor(self.codes[idx])
        message = torch.tensor(self.messages[idx])
        return {
            "commit_hash": commit_hash,
            "code": code,
            'message': message,
            "labels": label
        }
    
def training_deep_learning(params, dg_cache_path):
    commit_path = f'{dg_cache_path}/dataset/{params.repo_name}/commit'
    dictionary_path = f'{commit_path}/dict.jsonl' if params.dictionary is None else params.dictionary
    train_set_path = f'{commit_path}/{params.model}_{params.repo_name}_train.jsonl' if params.commit_train_set is None else params.commit_train_set
    val_set_path = f'{commit_path}/{params.model}_{params.repo_name}_val.jsonl' if params.commit_val_set is None else params.commit_val_set
    if params.model == "simcom":
        model_save_path = f'{dg_cache_path}/save/{params.repo_name}/com.pth'
    else:
        model_save_path = f'{dg_cache_path}/save/{params.repo_name}/{params.model}.pth'

    # Init model
    model = init_model(params.model, params.repo_language, params.device)
    if params.from_pretrain:
        model.initialize()
    else:
        model.initialize(dictionary=dictionary_path)

    # Load dataset
    train_data = open_jsonl(train_set_path)

    if params.model == "simcom":
        val_data = open_jsonl(val_set_path)

    dict_msg, dict_code = model.message_dictionary, model.code_dictionary

    code_dataset = CustomDataset(train_data, dict_code, dict_msg, model.hyperparameters)
    code_dataloader = DataLoader(code_dataset, batch_size=model.hyperparameters['batch_size'])

    if params.model == "simcom":
        val_code_dataset = CustomDataset(val_data, dict_code, dict_msg, model.hyperparameters)
        val_code_dataloader = DataLoader(val_code_dataset, batch_size=model.hyperparameters['batch_size'])

    optimizer = torch.optim.Adam(model.get_parameters(), lr=params.learning_rate)
    criterion = nn.BCELoss()

    # Validate
    best_valid_score = 0
    smallest_loss = 1000000
    early_stop_count = 5
    start_epoch = 1
    total_loss = 0

    if params.from_pretrain:
        checkpoint = torch.load(model_save_path)  # Load the last saved checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_loss = checkpoint['loss']

    for epoch in range(start_epoch, params.epochs + 1):
        for batch in code_dataloader:
            # Extract data from DataLoader
            code = batch["code"].to(model.device)
            message = batch["message"].to(model.device)
            labels = batch["labels"].to(model.device)

            optimizer.zero_grad()

            # ---------------------- DefectGuard -------------------------------
            predict = model(message, code)
            # ------------------------------------------------------------------
            
            loss = criterion(predict, labels)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print(f'Training: Epoch {epoch} / {params.epochs} -- Total loss: {total_loss}')

        if params.model == "simcom":
            model.com.eval()
            with torch.no_grad():
                all_predict, all_label = [], []
                for batch in tqdm(val_code_dataloader):
                    # Extract data from DataLoader
                    code = batch["code"].to(params.device)
                    message = batch["message"].to(params.device)
                    labels = batch["labels"].to(params.device)

                    # Forward
                    predict = model(message, code)
                    all_predict += predict.cpu().detach().numpy().tolist()
                    all_label += labels.cpu().detach().numpy().tolist()

            auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
            auc_pc_score = auc_pc(all_label, all_predict)
            print('Valid data -- AUC-ROC score:', auc_score,  ' -- AUC-PC score:', auc_pc_score)

            valid_score = auc_pc_score
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                print('Save a better model', best_valid_score.item())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, model_save_path)
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break
        else:
            loss_score = total_loss.item()
            print(loss_score < smallest_loss, loss_score, smallest_loss)
            if loss_score < smallest_loss:
                smallest_loss = loss_score
                print('Save a better model', smallest_loss)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, model_save_path)
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break

def training_machine_learning(params, dg_cache_path):
    train_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/feature/{params.repo_name}_train.jsonl' if params.feature_train_set is None else params.feature_train_set
    train_df = pd.read_json(train_df_path, lines=True)
    model = init_model(params.model, params.repo_language, params.device)

    cols = (
        ["la"]
        if model.model_name == "lapredict"
        else [
            "ns",
            "nd",
            "nf",
            "entropy",
            "la",
            "ld",
            "lt",
            "fix",
            "ndev",
            "age",
            "nuc",
            "exp",
            "rexp",
            "sexp",
        ]
    )
    X_train = train_df.loc[:, cols]
    y_train = train_df.loc[:, "label"]

    if model.model_name == "simcom":
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
        model.sim.fit(X_train, y_train)
        model.save_sim(f'{dg_cache_path}/save/{params.repo_name}')
    elif model.model_name == "lapredict" or model.model_name == "lr":
        model.model = sk_LogisticRegression(class_weight='balanced', max_iter=1000)
        model.model.fit(X_train, y_train)
        model.save(f'{dg_cache_path}/save/{params.repo_name}')
    elif model.model_name == "tlel":
        model.model.fit(X_train, y_train)
        model.save(f'{dg_cache_path}/save/{params.repo_name}')
        pass
    else:
        raise Exception("No such model")

def training(params):
    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))

    if params.model in ["deepjit", "simcom"]:
        training_deep_learning(params, dg_cache_path)

    if params.model in ["lapredict", "lr", "tlel", "simcom"]:
        training_machine_learning(params, dg_cache_path)

    

    