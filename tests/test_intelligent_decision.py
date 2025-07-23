import pandas as pd
from src.analysis.intelligent_decision import rank_candidates

def test_rank_candidates(tmp_path):
    # 构造最小数据集
    df = pd.DataFrame({
        "Material": ["A", "B"],
        "Dopant": ["Mg", "Ca"],
        "Concentration": [0.01, 0.02],
        "Position": ["Li-site", "Li-site"],
        "Predicted_Conductivity": [0.5, 0.4],
        "Predicted_Stability": [0.9, 0.95],
        "Uncertainty_Conductivity": [0.05, 0.02],
        "Uncertainty_Stability": [0.03, 0.04],
    })
    csv = tmp_path / "pred.csv"
    df.to_csv(csv, index=False)
    ranked = rank_candidates(csv, top_n=1)
    assert ranked.iloc[0]["Material"] == "A" 