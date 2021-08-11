# -*- coding: UTF-8 -*-
__author__ = "Shane Kao"
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

pd.options.mode.chained_assignment = None


class HourTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["hour_"] = X["hour"].apply(lambda x: str(x)[-2:])
        return X


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline, FeatureUnion
    hour_transformer = HourTransformer()
    train = pd.read_csv("raw_data/train.gz", nrows=10000000)
    train = hour_transformer.transform(train)

    from features.count_encoder import count
    from features.target_encoder import target
    from features.woe_encoder import woe
    from features.onehot_encoder import onehot
    from features.loo_encoder import loo
    from features.catboost_encoder import cat
    feature_union = FeatureUnion(
        transformer_list=[
            ("target", target),
            ("count", count),
            ("cat", cat),
            ("woe", woe),
            ("loo", loo),
            # ("onehot", onehot),
        ],
        transformer_weights=None
    )
    print(feature_union.fit_transform(X=train, y=train['click']).shape)

