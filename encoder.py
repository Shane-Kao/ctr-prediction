# -*- coding: UTF-8 -*-
__author__ = "Shane Kao"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from category_encoders.woe import WOEEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.count import CountEncoder

ENCODER_DICT = {
    "woe": WOEEncoder(handle_missing=0),
    "catboost": CatBoostEncoder(return_df=False),
    "loo": LeaveOneOutEncoder(return_df=False),
    "target": TargetEncoder(return_df=False),
    "count": CountEncoder(
        handle_unknown=0,
        combine_min_nan_groups=False,
        min_group_size=10
    ),
}


def create_encoder(encoder):
    function_transformer = FunctionTransformer(
        lambda x: x[
            [
                'site_id', 'site_domain', 'app_id', 'device_id', 'device_ip',
                'device_model', 'C14', 'C1', 'banner_pos', 'device_type',
                'device_conn_type', 'C15', 'C16', 'C18', 'site_category',
                'C19', 'C21', 'app_category', 'C20', 'C17', 'app_domain',
                'hour_']
        ]
    )
    pipeline = Pipeline(steps=[
        ("function_transformer", function_transformer),
        ("encoder", ENCODER_DICT[encoder]),
    ])
    return pipeline
