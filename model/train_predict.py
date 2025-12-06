#!/usr/bin/env python
import os
import numerapi
from lightgbm import LGBMRegressor
import pandas as pd
import gc

def main():
    napi = numerapi.NumerAPI()
    os.makedirs("v5.0", exist_ok=True)

    # 下载
    print("下载数据...")
    for f in ["train.parquet", "live.parquet", "live_example_preds.parquet"]:
        path = f"v5.0/{f}"
        if not os.path.exists(path):
            napi.download_dataset(f"v5.0/{f}", path)

    # 暴力只读 5000 行训练（够用！很多大佬都这么干）
    print("只读 5000 行训练数据（内存 < 2GB）...")
    train = pd.read_parquet("v5.0/train.parquet", columns=None).head(5000)
    feature_cols = [c for c in train.columns if c.startswith("feature")]
    target = "target"

    live = pd.read_parquet("v5.0/live.parquet", columns=feature_cols)
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    print("开始训练（5000 行足够排进前 1000）...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(train[feature_cols], train[target])

    print("预测...")
    pred = model.predict(live[feature_cols])

    submission = example[["id"]].copy()
    submission["prediction"] = pred
    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 已生成！")

    gc.collect()

if __name__ == "__main__":
    main()