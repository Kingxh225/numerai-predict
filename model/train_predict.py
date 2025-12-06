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

    print("读取 5000 行训练数据（只保留 feature + target）...")
    # 关键：先读列名，再只读 feature + target 列，彻底避开 id 列
    all_cols = pd.read_parquet("v5.0/train.parquet", columns=None).columns
    feature_cols = [c for c in all_cols if c.startswith("feature")]
    train = pd.read_parquet("v5.0/train.parquet", columns=feature_cols + ["target"]).head(5000)

    live = pd.read_parquet("v5.0/live.parquet", columns=feature_cols)
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    print("训练中...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(train[feature_cols], train["target"])

    print("预测 + 生成提交文件...")
    pred = model.predict(live[feature_cols])
    submission = example[["id"]].copy()
    submission["prediction"] = pred
    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 已生成，可以提交了！")

if __name__ == "__main__":
    main()