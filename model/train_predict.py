#!/usr/bin/env python
import os
import pandas as pd
import numerapi
from lightgbm import LGBMRegressor
import gc

def main():
    napi = numerapi.NumerAPI()
    os.makedirs("v5.0", exist_ok=True)

    print("正在下载数据...")
    for f in ["train.parquet", "validation.parquet", "live.parquet", "live_example_preds.parquet"]:
        path = f"v5.0/{f}"
        if not os.path.exists(path):
            napi.download_dataset(f"v5.0/{f}", path)

    print("读取特征列名...")
    feature_cols = [c for c in pd.read_parquet("v5.0/train.parquet", columns=None).columns 
                    if c.startswith("feature")]

    print("低内存读取训练数据（只读必要列 + float32 降精度）...")
    train = pd.read_parquet(
        "v5.0/train.parquet",
        columns=feature_cols + ["target"]
    ).astype("float32")          # 关键！降为 float32
    train["target"] = train["target"].astype("float32")

    live = pd.read_parquet("v5.0/live.parquet", columns=feature_cols).astype("float32")
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    print("开始训练（内存峰值 ≈2.8GB）...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        subsample=0.8,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True      # 强制列式存储，极大降低内存
    )
    model.fit(train[feature_cols], train["target"])

    print("预测中...")
    pred = model.predict(live[feature_cols])

    submission = example[["id"]].copy()
    submission["prediction"] = pred
    submission["prediction"].fillna(0.5, inplace=True)
    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 生成完毕！")

    # 清理内存
    del train, live, model, pred
    gc.collect()

if __name__ == "__main__":
    main()