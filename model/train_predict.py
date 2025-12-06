#!/usr/bin/env python
import os
import numerapi
from lightgbm import LGBMRegressor
import pandas as pd

def main():
    napi = numerapi.NumerAPI()
    os.makedirs("v5.0", exist_ok=True)

    print("下载数据...")
    for f in ["train.parquet", "live.parquet", "live_example_preds.parquet"]:
        path = f"v5.0/{f}"
        if not os.path.exists(path):
            napi.download_dataset(f"v5.0/{f}", path)

    print("读取训练数据（只取 5000 行）...")
    train_full = pd.read_parquet("v5.0/train.parquet")
    feature_cols = [c for c in train_full.columns if c.startswith("feature")]
    train = train_full[feature_cols + ["target"]].head(5000)

    print("读取 live 数据...")
    live = pd.read_parquet("v5.0/live.parquet")[feature_cols]
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    print("训练 + 预测...")
    model = LGBMRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=32, colsample_bytree=0.1, n_jobs=-1, verbosity=-1)
    model.fit(train[feature_cols], train["target"])
    pred = model.predict(live[feature_cols])

    # 关键：直接用 live 的 id（v5.0 正式 id 在 live.parquet 里）
    live_ids = pd.read_parquet("v5.0/live.parquet", columns=["id"])
    submission = pd.DataFrame({
        "id": live_ids["id"],
        "prediction": pred
    })

    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 已生成！可以提交了！")

if __name__ == "__main__":
    main()