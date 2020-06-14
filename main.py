import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import get_train_val_loaders
from model import TweetModel
from utils import loss_fn, compute_jaccard_score, EarlyStopping

warnings.filterwarnings('ignore')
import argparse
from test import test_it

paser = argparse.ArgumentParser()
paser.add_argument('--model', type=str, default='roberta-base')
paser.add_argument('--lr', type=float, default=3e-5)
paser.add_argument('--num_warmup_steps', type=int, default=400)
opt = paser.parse_args()


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed = 42
seed_everything(seed)
MODEL_PATH = opt.model
print(MODEL_PATH)


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename, scheduler):
    model.cuda()
    es = EarlyStopping(patience=2, mode="max")
    for epoch in range(num_epochs):
        flag = False
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0
            data_loader = dataloaders_dict[phase]
            tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
            for index, data in tk0:
                ids = data['ids'].cuda()
                masks = data['masks'].cuda()
                tweet = data['tweet']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].cuda()
                end_idx = data['end_idx'].cuda()
                length = data['length'].cuda()
                optimizer.zero_grad()
                step_jaccard = 0.0
                with torch.set_grad_enabled(phase == 'train'):

                    start_logits, end_logits, len_cls = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx, len_cls, length)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    epoch_loss += loss.item() * len(ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    len_cls = torch.softmax(len_cls, dim=1).cpu().detach().numpy()

                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i],
                            end_logits[i],
                            offsets[i],
                            len_cls[i],
                        )
                        epoch_jaccard += jaccard_score
                        step_jaccard += jaccard_score
                    tk0.set_postfix(stp='{}/{}'.format(index, len(data_loader)),
                                    loss='{:.4f}'.format(loss.item()),
                                    jaccard='{:.4f}'.format(step_jaccard / len(ids)))
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
            if phase == 'val':
                es(epoch_jaccard, model, model_path="type/" + MODEL_PATH + '/' + filename)
                if es.early_stop:
                    print("Early stopping")
                    flag = True
        if flag:
            break

    # torch.save(model.state_dict(), filename)


num_epochs = 5
batch_size = 16
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

train_df = pd.read_csv('data/train.csv')
train_df['text'] = train_df['text'].astype(str)
train_df['selected_text'] = train_df['selected_text'].astype(str)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
    print(f'Fold: {fold}')

    model = TweetModel(MODEL_PATH)
    num_train_steps = int(len(train_idx) / batch_size * num_epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=opt.num_warmup_steps,
        num_training_steps=num_train_steps
    )
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size, MODEL_PATH)

    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        f'roberta_fold{fold}.pth',
        scheduler
    )

test_it(MODEL_PATH)
