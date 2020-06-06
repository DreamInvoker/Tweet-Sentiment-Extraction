import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from dataset import get_train_val_loaders, get_test_loader
from model import TweetModel
from utils import loss_fn, get_selected_text, compute_jaccard_score, EarlyStopping

warnings.filterwarnings('ignore')


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
MODEL_PATH = 'roberta-base'


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
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

                optimizer.zero_grad()
                step_jaccard = 0.0
                with torch.set_grad_enabled(phase == 'train'):

                    start_logits, end_logits = model(ids, masks)

                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i],
                            end_logits[i],
                            offsets[i])
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
                es(epoch_jaccard, model, model_path=filename)
                if es.early_stop:
                    print("Early stopping")
                    flag = True
        if flag:
            break
    # torch.save(model.state_dict(), filename)


num_epochs = 3
batch_size = 32
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

train_df = pd.read_csv('data/train.csv')
train_df['text'] = train_df['text'].astype(str)
train_df['selected_text'] = train_df['selected_text'].astype(str)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
    print(f'Fold: {fold}')

    model = TweetModel(MODEL_PATH)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        f'roberta_fold{fold}.pth')


models = []
for t in os.listdir('type'):
    for model_file in os.listdir(os.path.join('type', t)):
        model = TweetModel(MODEL_PATH=t)
        model.cuda()
        model.load_state_dict(torch.load(model_file))
        model.eval()
        models.append(model)


test_df = pd.read_csv('data/test.csv')
test_df['text'] = test_df['text'].astype(str)
test_loader = get_test_loader(test_df)
predictions = []


for data in test_loader:
    ids = data['ids'].cuda()
    masks = data['masks'].cuda()
    tweet = data['tweet']
    offsets = data['offsets'].numpy()

    start_logits = []
    end_logits = []
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
            end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)
    for i in range(len(ids)):
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df['selected_text'] = predictions
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
sub_df.to_csv('submission.csv', index=False)
sub_df.head()
