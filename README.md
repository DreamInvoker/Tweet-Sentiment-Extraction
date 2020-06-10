# 推特情感短语抽取

# 加载预训练模型

- 下载预训练模型[roberta-base](https://cdn.huggingface.co/roberta-base-pytorch_model.bin)并重命名为pytorch_model.bin 存放至roberta-base目录中


# 训练
```shell script
./run.sh GPU_id
```

# 测试

-将得到的submission.csv提交至[kaggle比赛平台](https://www.kaggle.com/c/tweet-sentiment-extraction/submissions)