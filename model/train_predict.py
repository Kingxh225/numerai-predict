#!/usr/bin/env python
import os
import pandas as pd
import numerapi
from lightgbm import LGBMRegressor

def main():
    napi = numerapi.NumerAPI()
    
    # 创建目录（GitHub Actions 需要）
    os.makedirs("v5.0", exist_ok=True)
    
    # 下载最新数据
    print("正在下载数据...")
    napi.download_dataset("v5.0/live.parquet")
    napi.download_dataset("v5.0/train.parquet")
    napi.download_dataset("v5.0/validation.parquet")
    napi.download_dataset("v5.0/live_example_preds.parquet")

    # 读取数据
    train = pd.read_parquet("v5.0/train.parquet")
    live = pd.read_parquet("v5.0/live.parquet")
    example = pd.read_parquet("v5.0/live_example_preds.parquet")

    features = [c for c in train.columns if c.startswith("feature")]
    
    # 简单 LightGBM 模型（你可以后期换成自己的）
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        verbosity=-1
    )
    
    print("正在训练模型...")
    model.fit(train[features], train["target"])

    print("正在预测...")
    predictions = model.predict(live[features])

    submission = pd.DataFrame({
        "id": live["id"],
        "prediction": predictions
    })

    # 跟 example 保持一样顺序
    submission = example[["id"]].merge(submission, on="id", how="left")
    submission["prediction"].fillna(0.5, inplace=True)

    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv 已生成！")

if __name__ == "__main__":
    main()
