# -*- coding: UTF-8 -*-
__author__ = "Shane Kao"
import dill
import pandas as pd

test = pd.read_csv("./raw_data/test.gz")
submission = pd.read_csv("./raw_data/sampleSubmission.gz")

MODEL = "loo"
model, _, _ = dill.load(open("./model/{}.pkl".format(MODEL), 'rb'))

y_predict_proba = model.predict_proba(test)
y_predict_proba = y_predict_proba[:, 1]

submission["click"] = y_predict_proba

submission.to_csv("submission.csv", index=None)