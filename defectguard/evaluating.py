import os, torch, pickle
from torch.utils.data import Dataset, DataLoader
from .models import (
    DeepJIT,
    CC2Vec,
    SimCom,
    LAPredict,
    LogisticRegression,
    TLELModel as TLEL,
    JITLine,
)
from .utils.padding import padding_data
from .utils.logger import logger, logs
from .utils.utils import yield_jsonl, open_jsonl
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from datetime import datetime

def init_model(model_name, language, device):
    if model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif model_name == "cc2vec":
        return CC2Vec(language=language, device=device)
    elif model_name == "simcom":
        return SimCom(language=language, device=device)
    elif model_name == "lapredict":
        return LAPredict(language=language)
    elif model_name == "tlel":
        return TLEL(language=language)
    elif model_name == "jitline":
        return JITLine(language=language)
    elif model_name == "lr":
        return LogisticRegression(language=language)
    else:
        raise Exception("No such model")

class CustomDataset(Dataset):
    def __init__(self, data, code_dict, msg_dict, hyperparameters):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        commit_hash = self.data[0][idx]
        label = torch.tensor(self.data[3][idx], dtype=torch.float32)
        code = torch.tensor(self.data[1][idx])
        message = torch.tensor(self.data[2][idx])
        return {
            "commit_hash": commit_hash,
            "code": code,
            'message': message,
            "labels": label
        }
        
def load_dataset(file_path, hyperparameters, code_dict, msg_dict):
    data = read_json(file_path)
    commit_hashes = [d["commit_id"] for d in data]
    codes = [d["code_change"] for d in data]
    messages = [d["messages"] for d in data]
    labels = [d["label"] for d in data]
    del data
    padding_codes = padding_data(data=codes, dictionary=code_dict, params=hyperparameters, type='code')
    padding_msgs = padding_data(data=messages, dictionary=msg_dict, params=hyperparameters, type='msg')
    
    return (commit_hashes, codes, messages, labels)

def evaluating_deep_learning(pretrain, params, dg_cache_path):
    commit_path = f'{dg_cache_path}/dataset/{params.repo_name}/commit'
    dictionary_path = f'{commit_path}/{params.repo_name}_train_dict.pkl' if params.dictionary is None else params.dictionary
    test_set_path = f'{commit_path}/{params.model}_{params.repo_name}_test.pkl' if params.commit_test_set is None else params.commit_test_set
    pretrain_path = f'{dg_cache_path}/save/{params.repo_name}/{pretrain}'

    # Init model
    model = init_model(params.model, params.repo_language, params.device)

    if params.from_pretrain:
        model.initialize()
    else:
        model.initialize(dictionary=dictionary_path, state_dict=pretrain_path)

    # Load dataset
    test_data = load_dataset(test_set_path, model.hyperparameters, model.code_dictionary, model.message_dictionary)
    code_dataset = CustomDataset(test_data)
    code_dataloader = DataLoader(code_dataset, batch_size=1)

    if model.model_name == "simcom":
        model.com.eval()    
    else:
        model.model.eval()
    with torch.no_grad():
        commit_hashes, all_predict, all_label = [], [], []
        for batch in tqdm(code_dataloader):
            # Extract data from DataLoader
            commit_hashes.append(batch['commit_hash'][0])
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)

            # Forward
            predict = model(message, code)
            all_predict += predict.cpu().detach().numpy().tolist()
            all_label += labels.cpu().detach().numpy().tolist()
            
            # Free GPU memory
            del code, message, labels, predict
            torch.cuda.empty_cache()

    return commit_hashes, all_predict, all_label

def evaluating_machine_learning(pretrain, params, dg_cache_path):
    test_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/feature/{params.repo_name}_test.csv' if params.feature_test_set is None else params.feature_test_set
    test_df = pd.read_json(test_df_path, lines=True)
    model = init_model(params.model, params.repo_language, params.device)

    if params.from_pretrain:
        model.initialize()
    else:
        model.initialize(pretrain=f'{dg_cache_path}/save/{params.repo_name}/{pretrain}')

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
    commit_hashes = test_df.loc[:, "commit_id"].to_list()
    X_test = test_df.loc[:, cols]
    y_test = test_df.loc[:, "label"]

    y_proba = model.predict_proba(X_test)[:, 1]

    return commit_hashes, y_proba, y_test
        
def get_pretrain(model_name):
    if model_name == "deepjit":
        return "deepjit.pth"
    elif model_name == "sim":
        return "sim.pkl"
    elif model_name == "com":
        return "com.pth"
    elif model_name == "lapredict":
        return "lapredict.pkl"
    elif model_name == "lr":
        return "lr.pkl"
    elif model_name == "tlel":
        return "tlel.pkl"
    else:
        raise Exception("No such model")
    
def average(proba_1, proba_2):
    if len(proba_1) != len(proba_2):
        raise ValueError("Both lists must be of the same length")
    return [(x + y) / 2 for x, y in zip(proba_1, proba_2)]

def evaluating(params):
    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    predict_score_path = f'{dg_cache_path}/save/{params.repo_name}/predict_scores/'
    resutl_path = f'{dg_cache_path}/save/{params.repo_name}/results/'

    threshold = 0.5

    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))
    if os.path.isdir(predict_score_path) is False:
        os.makedirs(predict_score_path)
    if os.path.isdir(resutl_path) is False:
        os.makedirs(resutl_path)

    if params.model in ["deepjit", "simcom"]:
        model_name = params.model if params.model != "simcom" else "com"
        pretrain = get_pretrain(model_name)
        com_hashes, com_proba, com_ground_truth = evaluating_deep_learning(pretrain, params, dg_cache_path)
        sim_auc_score = roc_auc_score(y_true=com_ground_truth,  y_score=com_proba)
        com_pred = [1 if proba > threshold else 0 for proba in com_proba]
        sim_f1_score = f1_score(y_true=com_ground_truth, y_pred=com_pred)
        sim_accuracy = accuracy_score(y_true=com_ground_truth, y_pred=com_pred)
        sim_recall = recall_score(y_true=com_ground_truth, y_pred=com_pred)
        sim_precision = precision_score(y_true=com_ground_truth, y_pred=com_pred)

        logs(f'{dg_cache_path}/save/{params.repo_name}/results/auc.csv', params.repo_name, sim_auc_score, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/f1.csv', params.repo_name, sim_f1_score, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/acc.csv', params.repo_name, sim_accuracy, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/rc.csv', params.repo_name, sim_recall, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/prc.csv', params.repo_name, sim_precision, model_name)
        df = pd.DataFrame({'commit_hash': com_hashes, 'label': com_ground_truth, 'pred': com_proba})
        df.to_csv(f'{dg_cache_path}/save/{params.repo_name}/predict_scores/{model_name}.csv', index=False, sep=',')

    if params.model in ["lapredict", "lr", "tlel", "simcom"]:
        model_name = params.model if params.model != "simcom" else "sim"
        pretrain = get_pretrain(model_name)
        sim_hashes, sim_proba, sim_ground_truth = evaluating_machine_learning(pretrain, params, dg_cache_path)
        com_auc_score = roc_auc_score(y_true=sim_ground_truth,  y_score=sim_proba)
        sim_pred = [1 if proba > threshold else 0 for proba in sim_proba]
        com_f1_score = f1_score(y_true=sim_ground_truth, y_pred=sim_pred)
        com_accuracy = accuracy_score(y_true=sim_ground_truth, y_pred=sim_pred)
        com_recall = recall_score(y_true=sim_ground_truth, y_pred=sim_pred)
        com_precision = precision_score(y_true=sim_ground_truth, y_pred=sim_pred)

        logs(f'{dg_cache_path}/save/{params.repo_name}/results/auc.csv', params.repo_name, com_auc_score, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/f1.csv', params.repo_name, com_f1_score, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/acc.csv', params.repo_name, com_accuracy, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/rc.csv', params.repo_name, com_recall, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/prc.csv', params.repo_name, com_precision, model_name)
        df = pd.DataFrame({'commit_hash': sim_hashes, 'label': sim_ground_truth, 'pred': sim_proba})
        df.to_csv(f'{dg_cache_path}/save/{params.repo_name}/predict_scores/{model_name}.csv', index=False, sep=',')
    
    if params.model in ["simcom"]:
        assert com_hashes == sim_hashes
        simcom_proba = average(sim_proba, com_proba)
        auc_score = roc_auc_score(y_true=com_ground_truth,  y_score=simcom_proba)
        simcom_pred = [1 if proba > threshold else 0 for proba in simcom_proba]
        f1 = f1_score(y_true=com_ground_truth,  y_score=simcom_pred)
        acc = accuracy_score(y_true=com_ground_truth,  y_score=simcom_pred)
        rc = recall_score(y_true=com_ground_truth,  y_score=simcom_pred)
        prc = precision_score(y_true=com_ground_truth, y_pred=simcom_pred)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/auc.csv', params.repo_name, auc_score, params.model)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/f1.csv', params.repo_name, f1, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/acc.csv', params.repo_name, acc, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/rc.csv', params.repo_name, rc, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/prc.csv', params.repo_name, prc, model_name)