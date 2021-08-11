# -*- coding: UTF-8 -*-
__author__ = "Shane Kao"
import dill
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile

from preprocessing import HourTransformer


def create_model(X, y, encoder):
    pipeline = Pipeline([
        ('preprocessing', HourTransformer()),
        ('features', FeatureUnion(
            transformer_list=[
                ("encoder", encoder),
            ])),
        ('feature_selection', SelectPercentile()),
        ('model', LogisticRegression(solver='liblinear'))
    ])
    param = {
        'feature_selection__percentile': range(10, 110, 10),
        'model__fit_intercept': [True, False],
        'model__penalty': ['l1', 'l2'],
    }
    randomized_search_cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param,
        n_jobs=1,
        verbose=11,
        cv=TimeSeriesSplit(
            n_splits=3,
            test_size=int(X.shape[0]/10)
        ),
        scoring='neg_log_loss',
        refit=True,
        n_iter=5,
    )
    randomized_search_cv.fit(X, y)
    best_score = randomized_search_cv.best_score_
    best_parameters = randomized_search_cv.best_estimator_.get_params()
    return randomized_search_cv, best_score, best_parameters


if __name__ == '__main__':
    import pandas as pd

    from encoder import create_encoder
    from configs import TRAIN_SIZE, MODEL_LIST

    train = pd.read_csv("./raw_data/train.gz",
                        nrows=TRAIN_SIZE,
                        usecols=lambda x: x != "id")

    X_train = train
    y_train = train["click"]

    for encoder_name in MODEL_LIST:
        print(encoder_name)
        encoder = create_encoder(encoder_name)
        model, best_score, best_parameters = \
            create_model(X=X_train, y=y_train, encoder=encoder)
        dill.dump(
            obj=(model, best_score, best_parameters),
            file=open("./model/{}.pkl".format(encoder_name), 'wb')
        )
