#!/usr/bin/env python
import os
import numerapi
from lightgbm import LGBMRegressor
import pandas as pd

def main():
    napi = numerapi.NumerAPI()
    os.makedirs("v5.0", exist_ok=True)

    print("只下载必须的两个文件...")
    for f in ["train.parquet", "live.parquet"]:
        path = f"v5.0/{f}"
        if not os.path.exists(path):
            napi.download_dataset(f"v5.0/{f}", path)

    print("读取 5000 行训练数据...")
    train_full = pd.read_parquet("v5.0/train.parquet")
    feature_cols = [c for c in train_full.columns if c.startswith("feature")]
    train = train_full[feature_cols + ["target"]].head(5000)

    live = pd.read_parquet("v5.0/live.parquet")
    live_features = live[feature_cols]
    live_ids = live["id"]

    print("训练 + 预测...")
    model = LGBMRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=32, colsample_bytree=0.1, n_jobs=-1, verbosity=-1)
    model.fit(train[feature_cols], train["target"])
    pred = model.predict(live_features)

    pd.DataFrame({"id": live_ids, "prediction": pred}).to_csv("predictions.csv", index=False)
    print("predictions.csv 已生成！")

if __name__ == "__main__":
    main()