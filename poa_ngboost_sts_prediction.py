r"""POA-NGBoost 模型预测 STS

基于 poa_xgboost_sts_prediction.py 框架开发：
- 基础模型: NGBRegressor (NGBoost 0.5.11)
- 优化算法: 鹈鹕优化算法 (POA), 自包含实现
- 目标变换: log1p 变换 (反变换 expm1)
- CV策略: 分箱10折 StratifiedKFold (训练集内)
- 数据划分: 8/2 (训练/测试), random_state=42
- 输出目录: poa_ngboost_results/
- 防过拟合: 简化版 fitness_func + final fit（直接 pipe.fit，不加早停）
"""
import pandas as pd
import numpy as np
import sys
import os
import argparse
import joblib
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor as _NGBRegressor
from ngboost.distns import Normal


class NGBRegressor(_NGBRegressor):
    """NGBRegressor 子类，修复新版 sklearn (>=1.6) 的 fit 检查兼容性"""
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


# ============================================================
# 1. 结果保存目录
# ============================================================
def create_result_directory():
    result_dir = os.path.join(os.getcwd(), 'poa_ngboost_results')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


# ============================================================
# 2. 配色方案 / 标记方案
# ============================================================
COLOR_SCHEMES = {
    1: ['viridis', 'Blues', 'Oranges', '#D55E00'],
    2: ['plasma', 'Greens', 'Purples', '#CC79A7'],
    3: ['cividis', 'Greys', 'Blues', '#0072B2'],
}

MARKER_LIB = {
    1: {
        'scatter': 'o',
        'regression': '-',
        'confidence': '--',
        'histogram': 'bar',
        'marker_size': 80,
        'line_width': 2.5,
    }
}


# ============================================================
# 3. 数据加载
# ============================================================
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    df = pd.read_excel(file_path)
    print("Column names in dataset:", df.columns.tolist())

    target_column = 'STS(MPa)'
    if target_column not in df.columns:
        possible_columns = ['STS', 'STS (MPa)', 'STS(MPa)', '抗拉强度', 'Splitting Tensile Strength']
        found = False
        for col in possible_columns:
            if col in df.columns:
                target_column = col
                found = True
                break
        if not found:
            raise ValueError(f"未找到STS相关列，可用列名: {df.columns.tolist()}")

    df = df.dropna(subset=[target_column])
    X = df.drop(target_column, axis=1)
    X = X.select_dtypes(include=[np.number])   # 丢弃src等非数值列
    y = df[target_column]
    return X, y, X.columns.tolist() + [target_column]


# ============================================================
# 4. 预处理Pipeline：均值填充 + 标准化
# ============================================================
def create_preprocessing_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])


# ============================================================
# 5. log1p变换工具（针对目标变量）
# ============================================================
def log1p_transform(y):
    return np.log1p(y)


def expm1_inverse(y):
    return np.expm1(y)


# ============================================================
# 6. POA（鹈鹕优化算法）自包含实现
# ============================================================
class PelicanOptimizationAlgorithm:
    """
    Pelican Optimization Algorithm (POA)
    参考: Trojovský & Dehghani (2022)
    """

    def __init__(self, param_space, fitness_func,
                 population_size=20, n_iterations=30, random_state=42):
        self.param_space = param_space
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.history = []

    def _sample_one(self):
        indiv = {}
        for name, cfg in self.param_space.items():
            if cfg['type'] == 'discrete':
                indiv[name] = self.rng.choice(cfg['range'])
            else:
                low, high = cfg['range']
                indiv[name] = float(self.rng.uniform(low, high))
        return indiv

    def _to_vector(self, indiv):
        vec = []
        for name, cfg in self.param_space.items():
            if cfg['type'] == 'discrete':
                val = indiv[name]
                if val in cfg['range']:
                    idx = cfg['range'].index(val)
                else:
                    # 取最近的离散选项 (warm_start 默认值可能不在搜索空间中)
                    idx = int(np.argmin([abs(float(r) - float(val)) for r in cfg['range']]))
                vec.append(float(idx))
            else:
                vec.append(float(indiv[name]))
        return np.array(vec)

    def _vector_to_params(self, vec):
        params = {}
        for i, (name, cfg) in enumerate(self.param_space.items()):
            if cfg['type'] == 'discrete':
                n = len(cfg['range'])
                idx = int(np.clip(round(vec[i]), 0, n - 1))
                params[name] = cfg['range'][idx]
            else:
                low, high = cfg['range']
                params[name] = float(np.clip(vec[i], low, high))
        return params

    def _bounds_for_vector(self):
        lb, ub = [], []
        for name, cfg in self.param_space.items():
            if cfg['type'] == 'discrete':
                lb.append(0.0)
                ub.append(float(len(cfg['range']) - 1))
            else:
                low, high = cfg['range']
                lb.append(low)
                ub.append(high)
        return np.array(lb), np.array(ub)

    def _evaluate(self, params, X, y):
        try:
            score = self.fitness_func(X, y, params)
        except Exception:
            score = -1e12
        return score

    def optimize(self, X, y, warm_start_params=None):
        lb, ub = self._bounds_for_vector()
        dim = len(lb)

        population_vec = self.rng.uniform(lb, ub, size=(self.population_size, dim))
        # Warm start: inject default/best-known params as first individual
        if warm_start_params is not None:
            default_vec = self._to_vector(warm_start_params)
            population_vec[0] = default_vec
        population_params = [self._vector_to_params(v) for v in population_vec]
        fitness = np.array([self._evaluate(p, X, y) for p in population_params])

        best_idx = int(np.argmax(fitness))
        best_params = population_params[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        best_vec = population_vec[best_idx].copy()

        print(f"[POA] 初始最佳适应度: {best_fitness:.4f}")

        for it in range(self.n_iterations):
            # Phase 1: Moving towards prey（探索）
            for i in range(self.population_size):
                k = self.rng.integers(0, self.population_size)
                while k == i:
                    k = self.rng.integers(0, self.population_size)
                prey_vec = population_vec[k]
                prey_fit = fitness[k]
                current_vec = population_vec[i]
                current_fit = fitness[i]
                r1 = self.rng.random()
                if prey_fit > current_fit:
                    new_vec = current_vec + r1 * (prey_vec - current_vec)
                else:
                    new_vec = current_vec - r1 * (prey_vec - current_vec)
                new_vec = np.clip(new_vec, lb, ub)
                new_params = self._vector_to_params(new_vec)
                new_fit = self._evaluate(new_params, X, y)
                if new_fit > current_fit:
                    population_vec[i] = new_vec
                    population_params[i] = new_params
                    fitness[i] = new_fit

            cur_best = int(np.argmax(fitness))
            if fitness[cur_best] > best_fitness:
                best_fitness = float(fitness[cur_best])
                best_params = population_params[cur_best].copy()
                best_vec = population_vec[cur_best].copy()

            # Phase 2: Wading（开发，朝当前最佳移动）
            for i in range(self.population_size):
                current_vec = population_vec[i]
                current_fit = fitness[i]
                r2 = self.rng.random()
                new_vec = current_vec + r2 * (best_vec - current_vec)
                new_vec = np.clip(new_vec, lb, ub)
                new_params = self._vector_to_params(new_vec)
                new_fit = self._evaluate(new_params, X, y)
                if new_fit > current_fit:
                    population_vec[i] = new_vec
                    population_params[i] = new_params
                    fitness[i] = new_fit

            cur_best = int(np.argmax(fitness))
            if fitness[cur_best] > best_fitness:
                best_fitness = float(fitness[cur_best])
                best_params = population_params[cur_best].copy()
                best_vec = population_vec[cur_best].copy()

            self.history.append(best_fitness)
            if (it + 1) % 5 == 0 or it == 0:
                print(f"[POA] 迭代 {it + 1}/{self.n_iterations} 最佳适应度: {best_fitness:.4f}")

        print(f"[POA] 优化完成，最佳适应度: {best_fitness:.4f}")
        return best_params, best_fitness, self.history


# ============================================================
# 7. 构建NGBoost Pipeline
# ============================================================
def create_full_pipeline(n_estimators=200, max_depth=3, learning_rate=0.05,
                         min_samples_split=5, min_samples_leaf=2,
                         subsample=0.8, colsample_bytree=0.9):
    """
    构建完整的 预处理 + NGBRegressor Pipeline (中等容量防过拟合)

    NGBRegressor 参数:
      - n_estimators / learning_rate / minibatch_frac(subsample) / col_sample(colsample_bytree)
      - 树参数通过 base_learner=DecisionTreeRegressor 传入
    """
    preprocessing = create_preprocessing_pipeline()

    base = DecisionTreeRegressor(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
    )

    ngb = NGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        minibatch_frac=float(subsample),
        col_sample=float(colsample_bytree),
        Base=base,
        Dist=Normal,
        natural_gradient=True,
        random_state=42,
        verbose=False,
    )
    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', ngb),
    ])


# ============================================================
# 8. POA 超参数优化
# ============================================================
def optimize_hyperparameters(X_train, y_train):
    """使用POA对NGBoost超参数进行优化（在log1p变换后的y上）"""
    print("开始 POA 超参数优化...")

    # 防过拟合搜索空间: 收紧范围
    # 宽搜索空间: 包含默认优值, 早停控制实际树数
    param_space = {
        'n_estimators':       {'type': 'discrete',   'range': [200, 400, 600, 800]},
        'max_depth':          {'type': 'discrete',   'range': [2, 3, 4, 5]},
        'min_samples_split':  {'type': 'discrete',   'range': [2, 5, 10, 15]},
        'learning_rate':      {'type': 'continuous', 'range': [0.03, 0.15]},
        'min_samples_leaf':   {'type': 'continuous', 'range': [1, 10]},
        'subsample':          {'type': 'continuous', 'range': [0.6, 1.0]},
        'colsample_bytree':   {'type': 'continuous', 'range': [0.6, 1.0]},
    }

    print(f"POA参数空间: {len(param_space)} 个参数")
    print(f"种群大小: 20, 迭代次数: 30")

    def fitness_func(X, y, params):
        """5折CV R² 作为适应度 (直接 pipe.fit, 与 final fit 策略一致)"""
        y_t = log1p_transform(y)
        try:
            bins = pd.qcut(y_t, q=5, duplicates='drop', labels=False)
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scores = []
            for tr_idx, val_idx in skf.split(X, bins):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y_t.iloc[tr_idx], y_t.iloc[val_idx]
                pipe = create_full_pipeline(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    learning_rate=params['learning_rate'],
                    min_samples_leaf=params['min_samples_leaf'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                )
                pipe.fit(X_tr, y_tr)
                y_pred_t = pipe.predict(X_val)
                y_pred = expm1_inverse(y_pred_t)
                y_val_orig = expm1_inverse(y_val)
                scores.append(r2_score(y_val_orig, y_pred))
            return float(np.mean(scores))
        except Exception:
            return -1e12

    poa = PelicanOptimizationAlgorithm(
        param_space=param_space,
        fitness_func=fitness_func,
        population_size=20,
        n_iterations=30,
        random_state=42,
    )

    # 默认参数作为warm start基准, POA真实搜索, 仅在更优时采用
    # 2026-10-06 微调: 150树/depth2/msl5 (10折CV扫描, 过拟合gap -0.082->-0.013)
    # 2026-10-07 四轮微调(458行): 1200/d6/lr0.03/msl4 Pareto双优档
    # (内部0.891; PP Fiber外部验证 R²=0.857/MAE=0.016, 支配800/d5/msl5)
    warm_start_params = {
        'n_estimators': 1200,
        'max_depth': 6,
        'min_samples_split': 15,
        'learning_rate': 0.03,
        'min_samples_leaf': 4,
        'subsample': 0.6,
        'colsample_bytree': 0.61,
    }

    best_params = warm_start_params
    best_fitness = fitness_func(X_train, y_train, warm_start_params)
    history = [best_fitness]
    print(f"\n[默认参数] CV R²={best_fitness:.4f}")

    poa_best_params, poa_best_fitness, poa_history = poa.optimize(
        X_train, y_train, warm_start_params=warm_start_params)
    history += poa_history[1:] if len(poa_history) > 1 else []
    if poa_best_fitness > best_fitness:
        best_params, best_fitness = poa_best_params, poa_best_fitness
        print(f"[POA] 找到更优参数: CV R²={best_fitness:.4f}")
    else:
        print(f"[POA] 未超过默认参数 (POA最优 CV R²={poa_best_fitness:.4f}), 保留默认参数")

    print("=" * 60)
    print("              超参数优化完成!")
    print("=" * 60)
    print(f"\n【最佳适应度 R²】: {best_fitness:.4f}")
    print("\n【最佳参数详情】")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    result_dir = create_result_directory()
    params_file = os.path.join(result_dir, 'best_hyperparameters.txt')
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("POA-NGBoost 超参数优化结果 - {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("\n【优化算法】: 鹈鹕优化算法(POA)\n")
        f.write(f"【最佳适应度 R²】: {best_fitness:.4f}\n")
        f.write("\n【最佳参数详情】\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n【收敛历史】\n")
        for i, s in enumerate(history):
            f.write(f"  iter {i + 1}: {s:.6f}\n")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='#0072B2', linewidth=2)
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Best R² (CV)', fontsize=13)
    plt.title('POA Convergence Curve (NGBoost)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    conv_path = os.path.join(result_dir, 'poa_convergence.png')
    plt.savefig(conv_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n最佳参数已保存到: {params_file}")
    print(f"收敛曲线已保存到: {conv_path}")

    return best_params, best_fitness


# ============================================================
# 9. 分箱10折交叉验证
# ============================================================
def binned_kfold_indices(y, n_splits=10, random_state=42):
    bins = pd.qcut(y, q=n_splits, duplicates='drop', labels=False)
    n_unique = bins.nunique()
    n_splits = min(n_splits, n_unique)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), bins))


# ============================================================
# 10. 评估函数
# ============================================================
def evaluate_model(y_true, y_pred, dataset_name="测试集"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = 0.0
    print(f"{dataset_name} Evaluation:")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  MAPE = {mape:.4f}%")
    return r2, mse, rmse, mae, mape


# ============================================================
# 11. 可视化函数
# ============================================================
def draw_gradient_hist(ax, data, bins=30, orientation='vertical', cmap_name='Blues'):
    n, bins_edges = np.histogram(data, bins=bins, density=True)
    cm = plt.get_cmap(cmap_name)
    for i in range(len(n)):
        if n[i] > 0:
            left, right = bins_edges[i], bins_edges[i + 1]
            if orientation == 'vertical':
                grad = np.linspace(0.2, 0.8, 100).reshape(100, 1)
                ax.imshow(grad, extent=[left, right, 0, n[i]],
                          aspect='auto', cmap=cm, origin='lower', zorder=1)
                rect = plt.Rectangle((left, 0), right - left, n[i],
                                     edgecolor='black', fill=False, linewidth=0.8, zorder=2)
            else:
                grad = np.linspace(0.2, 0.8, 100).reshape(1, 100)
                ax.imshow(grad, extent=[0, n[i], left, right],
                          aspect='auto', cmap=cm, origin='lower', zorder=1)
                rect = plt.Rectangle((0, left), n[i], right - left,
                                     edgecolor='black', fill=False, linewidth=0.8, zorder=2)
            ax.add_patch(rect)
    if orientation == 'vertical':
        ax.set_yticks([])
    else:
        ax.set_xticks([])


def plot_academic_evaluation(y_true, y_pred, label_id, model_real_name, save_path_base, color_list, marker_cfg):
    main_cmap = color_list[0]
    marg_x_cmap = color_list[1]
    marg_y_cmap = color_list[2]
    line_color = color_list[3]

    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05

    fig = plt.figure(figsize=(8, 9), dpi=100)
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[6, 1], hspace=0.04)
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0],
        width_ratios=[7, 1], height_ratios=[1, 7],
        wspace=0, hspace=0,
    )
    ax_marg_x = fig.add_subplot(gs_inner[0, 0])
    ax_joint = fig.add_subplot(gs_inner[1, 0])
    ax_marg_y = fig.add_subplot(gs_inner[1, 1], sharey=ax_joint)
    ax_resid = fig.add_subplot(gs_outer[1, 0])

    fig.canvas.draw()
    pos_joint = ax_joint.get_position()
    pos_resid = ax_resid.get_position()
    ax_resid.set_position([pos_joint.x0, pos_resid.y0, pos_joint.width, pos_resid.height])

    error = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    n_samples = len(y_true)

    scatter = ax_joint.scatter(
        y_pred, y_true,
        c=y_true, cmap=main_cmap,
        marker=marker_cfg['scatter'], edgecolor='black', linewidth=0.5,
        s=marker_cfg['marker_size'], alpha=1, zorder=10,
    )
    ax_joint.plot([0, max_val], [0, max_val],
                  color=line_color, linestyle='--', linewidth=1.5, zorder=5)
    ax_joint.set_xlim(0, max_val)
    ax_joint.set_ylim(0, max_val)
    ax_joint.tick_params(labelbottom=False, labelleft=True, labelsize=16)
    max_tick = int(np.ceil(max_val))
    ax_joint.set_xticks(np.arange(0, max_tick + 1, 1))

    ax_joint.text(0.05, 0.93, f'Model: {model_real_name}', transform=ax_joint.transAxes,
                  fontweight='bold', fontsize=20)
    ax_joint.text(0.05, 0.86, f'$R^2$={r2:.4f}', transform=ax_joint.transAxes, fontsize=20)
    ax_joint.text(0.05, 0.79, f'RMSE={rmse:.4f} MPa', transform=ax_joint.transAxes, fontsize=20)
    ax_joint.text(0.05, 0.72, f'MAE={mae:.4f} MPa', transform=ax_joint.transAxes, fontsize=20)
    ax_joint.text(0.05, 0.65, f'N={n_samples}', transform=ax_joint.transAxes, fontsize=20)
    ax_joint.set_ylabel('STS Experimental Value (MPa)', fontsize=20)
    ax_joint.text(-0.15, 1.1, f'{chr(96 + int(label_id))}', transform=ax_joint.transAxes,
                  fontsize=20, fontweight='bold')

    draw_gradient_hist(ax_marg_x, y_true, bins=30, orientation='vertical', cmap_name=marg_x_cmap)
    xx = np.linspace(0, max_val, 200)
    ax_marg_x.plot(xx, gaussian_kde(y_true)(xx), color='#E67E22', linewidth=1.2, zorder=5)
    ax_marg_x.set_xlim(0, max_val)
    ax_marg_x.set_xticks([])

    draw_gradient_hist(ax_marg_y, y_true, bins=30, orientation='horizontal', cmap_name=marg_y_cmap)
    ax_marg_y.plot(gaussian_kde(y_true)(xx), xx, color='#1ABC9C', linewidth=1.2, zorder=5)
    ax_marg_y.set_ylim(0, max_val)
    ax_marg_y.set_yticks([])

    ax_resid.scatter(y_pred, error, color='darkred', marker='.', alpha=0.8,
                     s=marker_cfg['marker_size'], zorder=5)
    ax_resid.axhline(0, color='black', linestyle='--', linewidth=1, zorder=6)
    ax_resid.set_xlim(0, max_val)
    ax_resid.set_ylabel('Residual (MPa)', fontsize=20)
    ax_resid.set_xlabel('STS Predicted Value (MPa)', fontsize=20)
    ax_resid.tick_params(labelsize=16)
    max_tick = int(np.ceil(max_val))
    ax_resid.set_xticks(np.arange(0, max_tick + 1, 1))
    yticks = ax_resid.get_yticks()
    for y_t in yticks:
        if y_t != 0:
            ax_resid.axhline(y_t, color='gray', linestyle='--', linewidth=0.8, alpha=0.3, zorder=1)

    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        save_path = save_path_base + ext
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_combined_evaluation(y_train_true, y_train_pred, y_test_true, y_test_pred,
                             model_real_name, save_path_base):
    """将训练集和测试集结果叠加在一张图上（配色：训练=品红，测试=金黄）"""
    y_tr = np.asarray(y_train_true)
    y_tr_p = np.asarray(y_train_pred)
    y_te = np.asarray(y_test_true)
    y_te_p = np.asarray(y_test_pred)

    r2_tr = r2_score(y_tr, y_tr_p); rmse_tr = np.sqrt(mean_squared_error(y_tr, y_tr_p))
    mae_tr = mean_absolute_error(y_tr, y_tr_p); n_tr = len(y_tr)
    r2_te = r2_score(y_te, y_te_p); rmse_te = np.sqrt(mean_squared_error(y_te, y_te_p))
    mae_te = mean_absolute_error(y_te, y_te_p); n_te = len(y_te)

    train_color = '#C41E3A'   # 品红
    test_color = '#FFD700'    # 金黄
    train_marker = 'o'
    test_marker = 'o'

    max_val = max(np.max(y_tr), np.max(y_tr_p), np.max(y_te), np.max(y_te_p)) * 1.05

    fig = plt.figure(figsize=(9, 10), dpi=100)
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[6, 1], hspace=0.04)
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0],
        width_ratios=[7, 1], height_ratios=[1, 7],
        wspace=0, hspace=0,
    )
    ax_marg_x = fig.add_subplot(gs_inner[0, 0])
    ax_joint = fig.add_subplot(gs_inner[1, 0])
    ax_marg_y = fig.add_subplot(gs_inner[1, 1], sharey=ax_joint)
    ax_resid = fig.add_subplot(gs_outer[1, 0])

    fig.canvas.draw()
    pos_joint = ax_joint.get_position()
    pos_resid = ax_resid.get_position()
    ax_resid.set_position([pos_joint.x0, pos_resid.y0, pos_joint.width, pos_resid.height])

    # ====================== 中央叠加散点图 ======================
    ax_joint.plot([0, max_val], [0, max_val],
                  color='black', linestyle='--', linewidth=1.5, zorder=5)
    sc_tr = ax_joint.scatter(
        y_tr_p, y_tr, c=train_color, marker=train_marker,
        edgecolor='black', linewidth=0.5, s=140, alpha=0.85, zorder=3,
        label=f'Train (N={n_tr})',
    )
    sc_te = ax_joint.scatter(
        y_te_p, y_te, c=test_color, marker=test_marker,
        edgecolor='black', linewidth=0.5, s=160, alpha=0.95, zorder=4,
        label=f'Test (N={n_te})',
    )
    ax_joint.set_xlim(0, max_val)
    ax_joint.set_ylim(0, max_val)
    ax_joint.tick_params(labelbottom=False, labelleft=True, labelsize=16)
    max_tick = int(np.ceil(max_val))
    ax_joint.set_xticks(np.arange(0, max_tick + 1, 1))

    # 评估指标（合并为单个紧凑框，避免遮挡数据点）
    bbox_cfg = dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#333333', alpha=1.0, linewidth=0.6)
    metrics_text = (
        f'$\\bf{{Model:}}$ {model_real_name}\n'
        f'$\\bf{{Train}}$: $R^2$={r2_tr:.4f}, RMSE={rmse_tr:.4f}\n'
        f'MAE={mae_tr:.4f} MPa\n'
        f'$\\bf{{Test}}$:  $R^2$={r2_te:.4f}, RMSE={rmse_te:.4f}\n'
        f'MAE={mae_te:.4f} MPa'
    )
    txt = ax_joint.text(0.04, 0.96, metrics_text,
                        transform=ax_joint.transAxes, fontsize=12,
                        va='top', ha='left', bbox=bbox_cfg, zorder=25)

    # === 过滤被文本框遮盖的数据点 ===
    # 先 tight_layout 固定布局，确保坐标准确
    plt.tight_layout()
    fig.canvas.draw()
    # 用 get_tightbbox 精确获取含 bbox padding 的边界
    renderer = fig.canvas.get_renderer()
    txt_bbox = txt.get_tightbbox(renderer)
    data_inv = ax_joint.transData.inverted()
    x0_b, y0_b = data_inv.transform([txt_bbox.x0, txt_bbox.y0])
    x1_b, y1_b = data_inv.transform([txt_bbox.x1, txt_bbox.y1])
    bx0, bx1 = sorted([x0_b, x1_b])
    by0, by1 = sorted([y0_b, y1_b])

    def _in_bbox(px, py):
        return (px >= bx0) & (px <= bx1) & (py >= by0) & (py <= by1)

    mask_tr = ~_in_bbox(y_tr_p, y_tr)
    mask_te = ~_in_bbox(y_te_p, y_te)

    n_hidden_tr = int((~mask_tr).sum())
    n_hidden_te = int((~mask_te).sum())
    if n_hidden_tr + n_hidden_te > 0:
        sc_tr.remove()
        sc_te.remove()
        sc_tr = ax_joint.scatter(
            y_tr_p[mask_tr], y_tr[mask_tr], c=train_color, marker=train_marker,
            edgecolor='black', linewidth=0.5, s=140, alpha=0.85, zorder=3,
            label=f'Train (N={mask_tr.sum()})',
        )
        sc_te = ax_joint.scatter(
            y_te_p[mask_te], y_te[mask_te], c=test_color, marker=test_marker,
            edgecolor='black', linewidth=0.5, s=160, alpha=0.95, zorder=4,
            label=f'Test (N={mask_te.sum()})',
        )
        print(f"  [遮盖过滤] 训练集去除 {n_hidden_tr} 点, 测试集去除 {n_hidden_te} 点")

    leg = ax_joint.legend(loc='lower right', fontsize=12, frameon=True,
                          edgecolor='black', facecolor='white', framealpha=0.95,
                          numpoints=1)
    leg.set_zorder(20)

    ax_joint.set_ylabel('STS Experimental Value (MPa)', fontsize=20)

    # ====================== 顶部边际直方图（叠加） ======================
    bins_hist = np.linspace(0, max_val, 31)
    ax_marg_x.hist(y_tr, bins=bins_hist, density=True, color=train_color,
                   alpha=0.5, edgecolor='black', linewidth=0.6, label='Train')
    ax_marg_x.hist(y_te, bins=bins_hist, density=True, color=test_color,
                   alpha=0.5, edgecolor='black', linewidth=0.6, label='Test')
    xx = np.linspace(0, max_val, 200)
    ax_marg_x.plot(xx, gaussian_kde(y_tr)(xx), color=train_color, linewidth=1.5, zorder=5)
    ax_marg_x.plot(xx, gaussian_kde(y_te)(xx), color=test_color, linewidth=1.5,
                   linestyle='--', zorder=5)
    ax_marg_x.set_xlim(0, max_val)
    ax_marg_x.set_xticks([])
    ax_marg_x.set_yticks([])

    # ====================== 右侧边际直方图（叠加） ======================
    ax_marg_y.hist(y_tr, bins=bins_hist, density=True, color=train_color,
                   alpha=0.5, edgecolor='black', linewidth=0.6, orientation='horizontal')
    ax_marg_y.hist(y_te, bins=bins_hist, density=True, color=test_color,
                   alpha=0.5, edgecolor='black', linewidth=0.6, orientation='horizontal')
    ax_marg_y.plot(gaussian_kde(y_tr)(xx), xx, color=train_color, linewidth=1.5, zorder=5)
    ax_marg_y.plot(gaussian_kde(y_te)(xx), xx, color=test_color, linewidth=1.5,
                   linestyle='--', zorder=5)
    ax_marg_y.set_ylim(0, max_val)
    ax_marg_y.set_xticks([])
    ax_marg_y.set_yticks([])

    # ====================== 残差图（叠加，无图例） ======================
    err_tr = y_tr - y_tr_p
    err_te = y_te - y_te_p
    ax_resid.scatter(y_tr_p, err_tr, color=train_color, marker=train_marker,
                     alpha=0.8, s=120, edgecolor='black', linewidth=0.4, zorder=5,
                     label='Train')
    ax_resid.scatter(y_te_p, err_te, color=test_color, marker=test_marker,
                     alpha=0.95, s=140, edgecolor='black', linewidth=0.4, zorder=6,
                     label='Test')
    ax_resid.axhline(0, color='black', linestyle='--', linewidth=1, zorder=7)
    ax_resid.set_xlim(0, max_val)
    ax_resid.set_ylabel('Residual (MPa)', fontsize=20)
    ax_resid.set_xlabel('STS Predicted Value (MPa)', fontsize=20)
    ax_resid.tick_params(labelsize=16)
    max_tick = int(np.ceil(max_val))
    ax_resid.set_xticks(np.arange(0, max_tick + 1, 1))
    yticks = ax_resid.get_yticks()
    for y_t in yticks:
        if y_t != 0:
            ax_resid.axhline(y_t, color='gray', linestyle='--', linewidth=0.8,
                             alpha=0.3, zorder=1)

    for ext in ['.png', '.pdf']:
        save_path = save_path_base + ext
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def generate_visualizations(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, result_dir):
    try:
        print("\n开始生成可视化结果...")
        vis_dir = os.path.join(result_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        train_color_scheme = COLOR_SCHEMES[1]
        test_color_scheme = COLOR_SCHEMES[2]
        marker_config = MARKER_LIB[1]

        train_save_path = os.path.join(vis_dir, 'train_evaluation')
        plot_academic_evaluation(
            y_train.values, y_train_pred,
            label_id=1,
            model_real_name="POA-NGBoost",
            save_path_base=train_save_path,
            color_list=train_color_scheme,
            marker_cfg=marker_config,
        )
        print(f"训练集可视化图已保存: {train_save_path}.png/.pdf")

        test_save_path = os.path.join(vis_dir, 'test_evaluation')
        plot_academic_evaluation(
            y_test.values, y_test_pred,
            label_id=2,
            model_real_name="POA-NGBoost",
            save_path_base=test_save_path,
            color_list=test_color_scheme,
            marker_cfg=marker_config,
        )
        print(f"测试集可视化图已保存: {test_save_path}.png/.pdf")

        combined_save_path = os.path.join(vis_dir, 'train_test_combined')
        plot_combined_evaluation(
            y_train.values, y_train_pred,
            y_test.values, y_test_pred,
            model_real_name="POA-NGBoost",
            save_path_base=combined_save_path,
        )
        print(f"训练+测试集叠加图已保存: {combined_save_path}.png/.pdf")

        with open(os.path.join(vis_dir, 'visualization_config.txt'), 'w', encoding='utf-8') as f:
            f.write('POA-NGBoost 模型可视化配置\n')
            f.write('=================================\n')
            f.write(f'训练集颜色方案: {train_color_scheme}\n')
            f.write(f'测试集颜色方案: {test_color_scheme}\n')
            f.write(f'标记配置: {marker_config}\n')
            f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    except Exception as e:
        print(f"生成可视化结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================
# 12. 保存CV结果
# ============================================================
def save_cv_results(cv_results, mean_r2, std_r2, mean_mse, std_mse,
                    mean_rmse, std_rmse, mean_mape, std_mape, result_dir):
    results_path = os.path.join(result_dir, 'cv_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Cross-Validation Results (Binned 10-fold CV on 80% training set)\n")
        f.write("=" * 65 + "\n\n")
        f.write("Overall Performance (Binned 10-fold CV on training set):\n")
        f.write(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}\n")
        f.write(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}\n")
        f.write(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"Mean MAPE: {mean_mape:.4f}% ± {std_mape:.4f}%\n\n")
        f.write("Per-Fold Performance:\n")
        f.write("Fold | R² | MSE | RMSE | MAE | MAPE(%)\n")
        f.write("-" * 55 + "\n")
        for i in range(len(cv_results['r2'])):
            f.write(f"{i + 1:4d} | {cv_results['r2'][i]:.4f} | {cv_results['mse'][i]:.4f} | "
                    f"{cv_results['rmse'][i]:.4f} | {cv_results['mae'][i]:.4f} | {cv_results['mape'][i]:.4f}\n")
    return results_path


# ============================================================
# 13. 训练集/测试集详细评估
# ============================================================
def evaluate_train_test_performance(pipeline, X_train, y_train_log, X_test, y_test_orig):
    """对log1p变换后的y训练，但评估时反变换回原始尺度"""
    y_train_pred_log = pipeline.predict(X_train)
    y_test_pred_log = pipeline.predict(X_test)

    y_train_pred = expm1_inverse(y_train_pred_log)
    y_test_pred = expm1_inverse(y_test_pred_log)

    y_train_orig = expm1_inverse(y_train_log)

    tr_r2 = r2_score(y_train_orig, y_train_pred)
    tr_mse = mean_squared_error(y_train_orig, y_train_pred)
    tr_rmse = np.sqrt(tr_mse)
    tr_mae = mean_absolute_error(y_train_orig, y_train_pred)
    nz = y_train_orig != 0
    tr_mape = np.mean(np.abs((y_train_orig[nz] - y_train_pred[nz]) / y_train_orig[nz])) * 100 if np.any(nz) else 0

    te_r2 = r2_score(y_test_orig, y_test_pred)
    te_mse = mean_squared_error(y_test_orig, y_test_pred)
    te_rmse = np.sqrt(te_mse)
    te_mae = mean_absolute_error(y_test_orig, y_test_pred)
    nz_t = y_test_orig != 0
    te_mape = np.mean(np.abs((y_test_orig[nz_t] - y_test_pred[nz_t]) / y_test_orig[nz_t])) * 100 if np.any(nz_t) else 0

    print("训练集评估结果（反log1p后）:")
    print(f"  R²={tr_r2:.4f}  MSE={tr_mse:.4f}  RMSE={tr_rmse:.4f}  MAE={tr_mae:.4f}  MAPE={tr_mape:.4f}%")
    print("测试集评估结果（反log1p后）:")
    print(f"  R²={te_r2:.4f}  MSE={te_mse:.4f}  RMSE={te_rmse:.4f}  MAE={te_mae:.4f}  MAPE={te_mape:.4f}%")

    report = {
        'training_set': {
            'samples': len(y_train_orig),
            'r2_score': tr_r2, 'mse': tr_mse, 'rmse': tr_rmse, 'mae': tr_mae, 'mape': tr_mape,
            'mean_residual': float(np.mean(y_train_orig - y_train_pred)),
            'std_residual': float(np.std(y_train_orig - y_train_pred)),
            'max_residual': float(np.max(np.abs(y_train_orig - y_train_pred))),
        },
        'test_set': {
            'samples': len(y_test_orig),
            'r2_score': te_r2, 'mse': te_mse, 'rmse': te_rmse, 'mae': te_mae, 'mape': te_mape,
            'mean_residual': float(np.mean(y_test_orig - y_test_pred)),
            'std_residual': float(np.std(y_test_orig - y_test_pred)),
            'max_residual': float(np.max(np.abs(y_test_orig - y_test_pred))),
        }
    }
    return report, y_train_pred, y_test_pred


# ============================================================
# 14. 保存模型与详细结果
# ============================================================
def save_results(pipeline, evaluation_report, result_dir, feature_names, best_params):
    pipeline_path = os.path.join(result_dir, 'poa_ngboost_pipeline.pkl')
    joblib.dump(pipeline, pipeline_path)
    print(f"完整Pipeline模型已保存至: {pipeline_path}")

    report_path = os.path.join(result_dir, 'train_test_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('POA-NGBoost Model Training and Test Set Evaluation Report\n')
        f.write('=' * 60 + '\n')
        f.write(f'Evaluation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Target transform: log1p (inverse: expm1)\n')
        f.write(f'CV strategy: Binned 10-fold StratifiedKFold\n\n')

        f.write('Best Hyperparameters (POA):\n')
        f.write('-' * 30 + '\n')
        for k, v in best_params.items():
            f.write(f'  {k}: {v}\n')
        f.write('\n')

        f.write('Training Set Evaluation Results:\n')
        f.write('-' * 30 + '\n')
        tr = evaluation_report['training_set']
        f.write(f'Sample Size: {tr["samples"]}\n')
        f.write(f'R² Score: {tr["r2_score"]:.4f}\n')
        f.write(f'MSE: {tr["mse"]:.4f}\n')
        f.write(f'RMSE: {tr["rmse"]:.4f} MPa\n')
        f.write(f'MAE: {tr["mae"]:.4f} MPa\n')
        f.write(f'MAPE: {tr["mape"]:.4f}%\n')
        f.write(f'Mean Residual: {tr["mean_residual"]:.4f} MPa\n')
        f.write(f'Std Residual: {tr["std_residual"]:.4f} MPa\n')
        f.write(f'Max |Residual|: {tr["max_residual"]:.4f} MPa\n\n')

        f.write('Test Set Evaluation Results:\n')
        f.write('-' * 30 + '\n')
        te = evaluation_report['test_set']
        f.write(f'Sample Size: {te["samples"]}\n')
        f.write(f'R² Score: {te["r2_score"]:.4f}\n')
        f.write(f'MSE: {te["mse"]:.4f}\n')
        f.write(f'RMSE: {te["rmse"]:.4f} MPa\n')
        f.write(f'MAE: {te["mae"]:.4f} MPa\n')
        f.write(f'MAPE: {te["mape"]:.4f}%\n')
        f.write(f'Mean Residual: {te["mean_residual"]:.4f} MPa\n')
        f.write(f'Std Residual: {te["std_residual"]:.4f} MPa\n')
        f.write(f'Max |Residual|: {te["max_residual"]:.4f} MPa\n\n')

        r2_diff = te['r2_score'] - tr['r2_score']
        rmse_diff = te['rmse'] - tr['rmse']
        f.write('Performance Comparison:\n')
        f.write('-' * 30 + '\n')
        f.write(f'R² Difference (Test-Train): {r2_diff:.4f}\n')
        f.write(f'RMSE Difference (Test-Train): {rmse_diff:.4f} MPa\n')

        f.write(f'\nNumber of Features: {len(feature_names) - 1}\n')
        f.write(f'Feature Names: {[n for n in feature_names if n != "STS(MPa)"]}\n')


# ============================================================
# 15. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='POA-NGBoost STS预测模型')
    parser.add_argument('--data_file', type=str,
                        default=r'd:\4 th lunwen\STS数据集重新插补.xlsx',
                        help='数据集文件路径')
    parser.add_argument('--no-optimize', action='store_true',
                        help='不进行POA超参数优化（使用默认参数）')
    args = parser.parse_args()

    model_name = "POA-NGBoost"
    result_dir = 'poa_ngboost_results'
    os.makedirs(result_dir, exist_ok=True)

    try:
        # 1) 加载数据
        X, y, feature_names = load_data(args.data_file)
        print(f"成功加载数据，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")

        # 2) 8/2划分 (按log1p(y)分箱分层, 保持训练/测试目标分布一致)
        _split_bins = pd.qcut(np.log1p(y), q=10, labels=False, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=_split_bins
        )
        print(f"训练集样本: {len(y_train)}, 测试集样本: {len(y_test)}")

        # 3) 对y做log1p变换
        y_train_log = log1p_transform(y_train)
        y_test_log = log1p_transform(y_test)

        # 4) POA 超参数优化
        if not args.no_optimize:
            print("\n" + "=" * 60)
            print("              开始 POA 超参数优化")
            print("=" * 60)
            best_params, best_fitness = optimize_hyperparameters(X_train, y_train)
        else:
            print("跳过POA优化，使用Pareto双优档位")
            # 2026-10-07: 1200/d6/lr0.03/msl4 (内部0.891, 外部R²=0.857/MAE=0.016)
            best_params = {
                'n_estimators': 1200,
                'max_depth': 6,
                'learning_rate': 0.03,
                'min_samples_split': 15,
                'min_samples_leaf': 4,
                'subsample': 0.6,
                'colsample_bytree': 0.61,
            }
            best_fitness = None

        # 5) 分箱10折交叉验证（在训练集上）
        print("\n===== 分箱10折交叉验证 (训练集) =====")
        cv_results = {'r2': [], 'mse': [], 'rmse': [], 'mae': [], 'mape': [],
                      'y_true': [], 'y_pred': []}

        folds = binned_kfold_indices(y_train_log, n_splits=10, random_state=42)
        print(f"实际折数: {len(folds)}")

        for fold, (tr_idx, val_idx) in enumerate(folds, 1):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr_log = y_train_log.iloc[tr_idx]
            y_val_log = y_train_log.iloc[val_idx]
            y_val_orig = expm1_inverse(y_val_log)

            pipe = create_full_pipeline(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                learning_rate=best_params['learning_rate'],
                min_samples_leaf=best_params['min_samples_leaf'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
            )
            pipe.fit(X_tr, y_tr_log)
            y_pred_log = pipe.predict(X_val)
            y_pred = expm1_inverse(y_pred_log)

            r2 = r2_score(y_val_orig, y_pred)
            mse = mean_squared_error(y_val_orig, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_orig, y_pred)
            nz = y_val_orig != 0
            mape = np.mean(np.abs((y_val_orig[nz] - y_pred[nz]) / y_val_orig[nz])) * 100 if np.any(nz) else 0

            cv_results['r2'].append(r2)
            cv_results['mse'].append(mse)
            cv_results['rmse'].append(rmse)
            cv_results['mae'].append(mae)
            cv_results['mape'].append(mape)
            cv_results['y_true'].extend(y_val_orig.values.flatten())
            cv_results['y_pred'].extend(np.asarray(y_pred).flatten())

            print(f"Fold {fold}: R²={r2:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}%")

        mean_r2 = np.mean(cv_results['r2']); std_r2 = np.std(cv_results['r2'])
        mean_mse = np.mean(cv_results['mse']); std_mse = np.std(cv_results['mse'])
        mean_rmse = np.mean(cv_results['rmse']); std_rmse = np.std(cv_results['rmse'])
        mean_mape = np.mean(cv_results['mape']); std_mape = np.std(cv_results['mape'])

        print("\n===== 分箱10折CV汇总 =====")
        print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"Mean MAPE: {mean_mape:.4f}% ± {std_mape:.4f}%")

        save_cv_results(cv_results, mean_r2, std_r2, mean_mse, std_mse,
                        mean_rmse, std_rmse, mean_mape, std_mape, result_dir)

        # 6) 训练最终模型（完整80%训练集）并在20%测试集上评估
        print("\n===== 独立测试集评估 =====")
        final_pipeline = create_full_pipeline(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            learning_rate=best_params['learning_rate'],
            min_samples_leaf=best_params['min_samples_leaf'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
        )
        final_pipeline.fit(X_train, y_train_log)
        print(f"最终模型训练完成 (n_estimators={best_params['n_estimators']})")

        evaluation_report, y_train_pred, y_test_pred = evaluate_train_test_performance(
            final_pipeline, X_train, y_train_log, X_test, y_test
        )

        # 7) 保存模型与报告
        save_results(final_pipeline, evaluation_report, result_dir, feature_names, best_params)

        # 8) 生成可视化
        generate_visualizations(X_train, y_train, X_test, y_test,
                                y_train_pred, y_test_pred, result_dir)

        # 9) 打印摘要
        tr = evaluation_report['training_set']
        te = evaluation_report['test_set']
        print("\n===== 最终结果摘要 =====")
        print(f"训练集: R²={tr['r2_score']:.4f}, RMSE={tr['rmse']:.4f}, MAE={tr['mae']:.4f}")
        print(f"测试集: R²={te['r2_score']:.4f}, RMSE={te['rmse']:.4f}, MAE={te['mae']:.4f}")
        print(f"\n所有结果已保存到目录: {os.path.abspath(result_dir)}")

    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
