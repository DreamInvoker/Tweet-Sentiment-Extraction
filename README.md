# 推特情感短语抽取

# 加载预训练模型

- 下载预训练模型[roberta-base](https://cdn.huggingface.co/roberta-base-pytorch_model.bin)并重命名为pytorch_model.bin 存放至roberta-base目录中

# 环境依赖
- python3.6
- torch 1.4.0
- transformers 2.10.0
- scikit-learn 0.22.1 
- tqdm 4.42.1
- tokenizers 0.7.0
- numpy 1.18.1
- pandas 1.0.4
# 训练

use GPU:
```shell script
./run.sh GPU_id
```
use CPU:
```shell script
./run.sh -1
```
# 测试

- 将得到的submission.csv提交至[kaggle比赛平台](https://www.kaggle.com/c/tweet-sentiment-extraction/submissions)