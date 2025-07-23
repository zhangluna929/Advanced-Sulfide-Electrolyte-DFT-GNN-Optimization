"""intelligent_decision.py
智能决策模块：基于 GNN 预测结果与不确定性评估，输出下一批最值得 DFT 计算的候选结构。

用法示例：
>>> from src.analysis.intelligent_decision import rank_candidates
>>> df = rank_candidates('results/doping_predictions.csv', top_n=50)

输入 CSV 至少包含以下列：
    Material, Dopant, Concentration, Position,
    Predicted_Conductivity, Predicted_Stability,
    Uncertainty_Conductivity, Uncertainty_Stability

评分函数默认以
    Score = w_c * σ + w_s * Stab - w_u * (U_c + U_s)
其中 w_c=0.6, w_s=0.4, w_u=0.5。
可通过参数自定义权重。
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Union, Tuple


def _default_score(row: pd.Series, w: Tuple[float, float, float]) -> float:
    wc, ws, wu = w
    return wc * row["Predicted_Conductivity"] + ws * row["Predicted_Stability"] - wu * (
        row["Uncertainty_Conductivity"] + row["Uncertainty_Stability"]
    )


def rank_candidates(
    csv_path: Union[str, Path],
    top_n: int = 20,
    weights: Tuple[float, float, float] = (0.6, 0.4, 0.5),
    save_path: Union[str, Path] | None = None,
) -> pd.DataFrame:
    """读取预测结果并按加权得分排序，返回 Top-N。

    Parameters
    ----------
    csv_path : str or Path
        预测结果 CSV 路径。
    top_n : int, default 20
        返回的候选数目。
    weights : tuple, default (0.6, 0.4, 0.5)
        (conductivity_weight, stability_weight, uncertainty_weight)。
    save_path : str or Path, optional
        若提供，则将排序结果保存至该路径。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
        "Predicted_Conductivity",
        "Predicted_Stability",
        "Uncertainty_Conductivity",
        "Uncertainty_Stability",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.copy()
    df["Decision_Score"] = df.apply(_default_score, axis=1, args=(weights,))
    df_sorted = df.sort_values("Decision_Score", ascending=False).head(top_n)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_sorted.to_csv(save_path, index=False)

    return df_sorted 