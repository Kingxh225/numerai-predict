#!/usr/bin/env python
import os
import pandas as pd
import numerapi
from lightgbm import LGBMRegressor, Dataset
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
    # 只读第一行获取列名，避免全载
    sample_df = pd.read_parquet("v5.0/train.parquet", nrows=1)
    feature_cols = [c for c in sample_df.columns if c.startswith("feature")]

    print("低内存训练：用 LightGBM Dataset 分块加载...")
    # 关键！直接从 parquet 路径创建 Dataset，不转 pandas 全表
    train_ds = Dataset(
        f"v5.0/train.parquet",
        label="target",
        feature_name=feature_cols,
        params={"data_has_header": True, "enable_categorical": False}
    )

    live = pd.read_parquet("v5.0/live.parquet", columns=feature_cols)
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    print("开始训练（内存峰值 < 2GB）...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        subsample=0.8,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True
    )
    # 用 train_ds 训练，避开 pandas 内存坑
    model.fit(train_ds)

    print("预测中...")
    pred = model.predict(live[feature_cols])

    submission = example[["id"]].copy()
    submission["prediction"] = pred
    submission["prediction"].fillna(0.5, inplace=True)
    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 生成完毕！")

    # 清理
    del train_ds, live, model, pred, submission
    gc.collect()

if __name__ == "__main__":
    main()