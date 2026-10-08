# -*- coding: utf-8 -*-
"""POA-NGB 劈裂抗拉强度(STS)预测模型封装

- 加载工作区已训练好的 NGBoost 管线 (preprocessing + NGBRegressor)
- 目标在 log1p 空间训练, 预测经 expm1 反变换回 MPa
- NGBoost 输出预测分布, 可给定点估计与 95% 置信区间
"""
import os
import sys
import glob
import functools

import numpy as np
import pandas as pd
import joblib

# 工作区根目录: 本文件位于 <root>/sts_predictor/sts_predictor/core/
WORKSPACE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# 外部验证集里粒径写作 D10(um), 统一到训练特征名 D10(μm)
COLUMN_ALIASES = {"D10(um)": "D10(μm)"}


def _inject_ngb_class():
    """NGBoost 管线以 __main__.NGBRegressor 序列化, 加载前注入类定义"""
    if WORKSPACE not in sys.path:
        sys.path.insert(0, WORKSPACE)
    import poa_ngboost_sts_prediction as ngb_mod
    sys.modules["__main__"].NGBRegressor = ngb_mod.NGBRegressor


class STSPredictor(object):
    """POA-NGB 管线的薄封装"""

    def __init__(self, pkl_path=None):
        _inject_ngb_class()
        if pkl_path is None:
            hits = glob.glob(os.path.join(
                WORKSPACE, "poa_ngboost_results", "*.pkl"))
            if not hits:
                raise FileNotFoundError(
                    "未找到 poa_ngboost_results/*.pkl 模型文件")
            pkl_path = hits[0]
        self.pkl_path = pkl_path
        self.pipe = joblib.load(pkl_path)
        self.features = list(self.pipe.feature_names_in_)

    # ----------------------------------------------------------
    def _align(self, df):
        """列名归一化 + 按训练特征顺序对齐, 返回数值型 DataFrame"""
        data = df.rename(columns=COLUMN_ALIASES).copy()
        missing = [f for f in self.features if f not in data.columns]
        if missing:
            raise ValueError("输入数据缺少特征列: " + ", ".join(missing))
        out = data[self.features].apply(
            pd.to_numeric, errors="coerce")
        if out.isna().any().any():
            bad = out.columns[out.isna().any()].tolist()
            raise ValueError("以下特征存在空值或非数值: " + ", ".join(bad))
        return out

    # ----------------------------------------------------------
    def predict(self, df, ci=0.95):
        """批量预测

        返回与 df 等长的 DataFrame:
            Predicted STS (MPa), Predicted STS (kPa),
            CI Low (MPa), CI High (MPa)
        """
        X = self._align(df)
        log_pred = np.asarray(self.pipe.predict(X)).ravel()
        pred = np.clip(np.expm1(log_pred), 0.0, None)

        pre = self.pipe.named_steps["preprocessing"].transform(X)
        dist = self.pipe.named_steps["model"].pred_dist(pre)
        lo, hi = dist.interval(ci)
        lo = np.clip(np.expm1(np.asarray(lo).ravel()), 0.0, None)
        hi = np.clip(np.expm1(np.asarray(hi).ravel()), 0.0, None)

        return pd.DataFrame({
            "Predicted STS (MPa)": pred,
            "Predicted STS (kPa)": pred * 1000.0,
            "CI Low (MPa)": lo,
            "CI High (MPa)": hi,
        })

    # ----------------------------------------------------------
    def predict_one(self, values, ci=0.95, n_samples=20000, seed=42):
        """单点预测, 返回 (结果dict, MPa空间后验样本数组)"""
        if isinstance(values, dict):
            row = pd.DataFrame([values])
        else:
            row = pd.DataFrame([dict(zip(self.features, values))])
        res = self.predict(row, ci=ci).iloc[0].to_dict()

        X = self._align(row)
        pre = self.pipe.named_steps["preprocessing"].transform(X)
        dist = self.pipe.named_steps["model"].pred_dist(pre)
        rng = np.random.default_rng(seed)
        # ngboost Normal.sample, 单行返回 (n_samples,)
        try:
            samples = np.asarray(
                dist.sample(n_samples, rng=rng)).ravel()
        except TypeError:
            samples = np.asarray(dist.sample(n_samples)).ravel()
        samples = np.clip(np.expm1(samples), 0.0, None)
        return res, samples


@functools.lru_cache(maxsize=1)
def get_predictor(pkl_path=None):
    """进程级单例, 供 Streamlit 缓存复用"""
    return STSPredictor(pkl_path)
