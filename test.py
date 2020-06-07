import os

import numpy as np
import pandas as pd
import torch

from dataset import get_test_loader
from model import TweetModel
from utils import get_selected_text


def test(MODEL_PATH='roberta-base'):
    models = []
    for t in os.listdir('type'):
        for model_file in os.listdir(os.path.join('type', t)):
            model = TweetModel(MODEL_PATH=t)
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(os.path.join('type', t), model_file)))
            model.eval()
            models.append(model)

    test_df = pd.read_csv('data/test.csv')
    test_df['text'] = test_df['text'].astype(str)
    test_loader = get_test_loader(test_df, MODEL_PATH=MODEL_PATH)
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
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()
