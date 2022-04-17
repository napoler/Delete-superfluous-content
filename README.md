# Delete superfluous content 删除冗余信息

## 思路
https://www.kdocs.cn/view/l/clMszgx6DfBU?from=docs

- 准备文本数据
- 段落数据拆分为句子
- 五个一组组成数据
- 随机替换第三个句子，替换的为1，未替换的为0
- 是用cls判别模型训练二分类模型


## 依赖

> pip install -U jsonargparse[signatures] transformers torchmetrics pytorch-lightning
## 预处理数据
> python preconditioning.py
> 
## 生成数据
https://github.com/napoler/BulidDataset/blob/main/buildDataSequenceClassification.py



