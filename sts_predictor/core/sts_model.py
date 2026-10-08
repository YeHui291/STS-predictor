# -*- coding: utf-8 -*-
"""POA-NGB 劈裂抗拉强度(STS)预测模型封装 (自包含, 兼容 Streamlit Cloud)

- 部署包目录(本文件向上两级)内自带:
    poa_ngboost_sts_prediction.py        NGBRegressor 类定义
    poa_ngboost_results/*.pkl            已训练管线
    data/sts_train_data.xlsx             训练数据(用于强度分位)
- 本地开发时, 若部署包内缺文件, 自动回退到工作区根目录(向上三级)
- 目标在 log1p 空间训练, 预测经 expm1 反变换回 MPa
"""
import os
import sys
import glob
import importlib.util
import functools

import numpy as np
import pandas as pd
import joblib

# 本文件: <root>/sts_predictor/core/sts_model.py
_PKG_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.path.abspath(os.path.join(_PKG_CORE_DIR, "..", ".."))   # 部署包根
WORKSPACE_ROOT = os.path.abspath(os.path.join(_PKG_CORE_DIR, "..", "..", ".."))  # 本地工作区根

# 外部验证集里粒径写作 D10(um), 统一到训练特征名 D10(μm)
COLUMN_ALIASES = {"D10(um)": "D10(μm)"}

NGB_MODULE_NAME = "poa_ngboost_sts_prediction"
NGB_MODULE_FILE = NGB_MODULE_NAME + ".py"
PKL_REL = os.path.join("poa_ngboost_results", "*.pkl")
DATA_CANDIDATES = [
    os.path.join("data", "sts_train_data.xlsx"),
    "STS数据集重新插补.xlsx",
]


def candidate_roots():
    """按优先级返回资源根目录: 环境变量 > 部署包根 > 本地工作区根"""
    roots = []
    env_root = os.environ.get("STS_ROOT")
    if env_root:
        roots.append(os.path.abspath(env_root))
    roots.append(DEPLOY_ROOT)
    if WORKSPACE_ROOT != DEPLOY_ROOT:
        roots.append(WORKSPACE_ROOT)
    # 去重保序
    seen, uniq = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _find_first(rel_patterns):
    """在各候选根下查找第一个存在的文件, 返回绝对路径(找不到返回 None)"""
    if isinstance(rel_patterns, str):
        rel_patterns = [rel_patterns]
    for root in candidate_roots():
        for rel in rel_patterns:
            hits = glob.glob(os.path.join(root, rel))
            if hits:
                return hits[0]
    return None


def find_pkl():
    return _find_first(PKL_REL)


def find_train_data():
    return _find_first(DATA_CANDIDATES)


def _inject_ngb_class():
    """加载 NGBRegressor 类并注入 __main__ (pickle 以 __main__.NGBRegressor 序列化)

    优先从部署包内按文件路径加载, 避免对 sys.path / 工作目录的依赖。
    """
    if getattr(sys.modules.get("__main__"), "NGBRegressor", None) is not None:
        return

    mod = sys.modules.get(NGB_MODULE_NAME)
    if mod is None:
        mod_path = _find_first(NGB_MODULE_FILE)
        if mod_path is None:
            raise ModuleNotFoundError(
                "Cannot find %s. Make sure it is committed next to app.py "
                "in the deployment repository." % NGB_MODULE_FILE)
        spec = importlib.util.spec_from_file_location(NGB_MODULE_NAME, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[NGB_MODULE_NAME] = mod
        spec.loader.exec_module(mod)

    ngb_cls = getattr(mod, "NGBRegressor", None)
    if ngb_cls is None:
        raise AttributeError(
            "NGBRegressor class not found in %s" % NGB_MODULE_FILE)
    sys.modules["__main__"].NGBRegressor = ngb_cls


class STSPredictor(object):
    """POA-NGB 管线的薄封装"""

    def __init__(self, pkl_path=None):
        _inject_ngb_class()
        if pkl_path is None:
            pkl_path = find_pkl()
        if not pkl_path or not os.path.exists(pkl_path):
            raise FileNotFoundError(
                "Trained pipeline not found (poa_ngboost_results/*.pkl). "
                "Commit the .pkl file into the deployment repository.")
        self.pkl_path = pkl_path
        self.pipe = joblib.load(pkl_path)
        self.features = list(self.pipe.feature_names_in_)
        self.data_path = find_train_data()

    # ----------------------------------------------------------
    def _align(self, df):
        """列名归一化 + 按训练特征顺序对齐, 返回数值型 DataFrame"""
        data = df.rename(columns=COLUMN_ALIASES).copy()
        missing = [f for f in self.features if f not in data.columns]
        if missing:
            raise ValueError("Input is missing feature columns: " + ", ".join(missing))
        out = data[self.features].apply(pd.to_numeric, errors="coerce")
        if out.isna().any().any():
            bad = out.columns[out.isna().any()].tolist()
            raise ValueError("Non-numeric or empty values in: " + ", ".join(bad))
        return out

    # ----------------------------------------------------------
    def predict(self, df, ci=0.95):
        """批量预测, 返回点估计(MPa/kPa)与 95% 置信区间"""
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
        try:
            samples = np.asarray(dist.sample(n_samples, rng=rng)).ravel()
        except TypeError:
            samples = np.asarray(dist.sample(n_samples)).ravel()
        samples = np.clip(np.expm1(samples), 0.0, None)
        return res, samples


@functools.lru_cache(maxsize=1)
def get_predictor(pkl_path=None):
    """进程级单例, 供 Streamlit 缓存复用"""
    return STSPredictor(pkl_path)
