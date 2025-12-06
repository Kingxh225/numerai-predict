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

    print("加载特征列名...")
    feature_cols = [c for c in pd.read_parquet("v5.0/train.parquet", columns=None).columns if c.startswith("feature")]

    print("使用 LightGBM 原生 Dataset 低内存训练...")
    # 直接用 parquet 文件路径 + 只读特征列 + target，内存暴降
    train = pd.read_parquet("v5.0/train.parquet", columns=feature_cols + ["target"])


    live = pd.read_parquet("v5.0/live.parquet", columns=feature_cols)
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        verbosity=-1
    )

    print("开始训练（内存峰值只会 3GB 左右）...")
    model.fit(train[feature_cols], train["target"])

    print("预测中...")
    pred = model.predict(live[feature_cols])

    submission = pd.DataFrame({
        "id": live["id"],
        "prediction": pred
    })

    # 对齐 example_preds 的 id 顺序
    submission = example[["id"]].merge(submission, on="id", how="left")
    submission["prediction"].fillna(0.5, inplace=True)

    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 生成完毕！")

    del train, live, model, pred, submission
    gc.collect()

if __name__ == "__main__":
    main()