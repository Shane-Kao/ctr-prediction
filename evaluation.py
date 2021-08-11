# -*- coding: UTF-8 -*-
__author__ = "Shane Kao"
import dill
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from configs import TRAIN_SIZE, MODEL_LIST

train = pd.read_csv("./raw_data/train.gz", usecols=lambda x: x != "id")

_, X_test, _, y_test = train_test_split(
    train, train['click'], train_size=TRAIN_SIZE, random_state=42,
    shuffle=False
)

del train
print(X_test.shape)

for encoder_name in MODEL_LIST:
    model, best_score_, best_parameters = \
            dill.load(open("./model/{}.pkl".format(encoder_name), 'rb'))
    y_predict_proba = model.predict_proba(X_test)
    y_predict_proba = y_predict_proba[:, 1]
    log_loss_ = log_loss(y_true=y_test, y_pred=y_predict_proba)
    print(encoder_name, best_score_, log_loss_,
            best_parameters['feature_selection__percentile'],
            best_parameters['model__fit_intercept'],
            best_parameters['model__penalty'],)
