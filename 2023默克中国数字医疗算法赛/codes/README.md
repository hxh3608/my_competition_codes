第一次正式打数据科学比赛，在这个比赛感谢各位大佬的分享，自己从中学到了很多东西。谢谢！！！
最终排名：Top10
以下分享下自己在基于asir大佬（官方baseline）做的一些工作。

# 1. 赛事简介

赛题链接：[2023数字医疗算法应用创新大赛-食物与疾病关系预测算法赛道](https://www.heywhale.com/home/competition/63eee2950644cee838881588/content/0)

## 赛题背景

越来越多的证据表明，食物分子与慢性疾病之间存在关联甚至治疗关系。营养成分可能直接或间接地作用于人类基因组，并调节参与疾病风险和疾病进展的多个过程。一般来说，营养物质是为活动提供能量的物质，是身体生长和修复的物质，是保持免疫系统健康的物质。随着生物医学数据量的爆炸式增长，现在有可能通过数据驱动的方法通过化合物建立疾病和食物之间的联系，并探索食物营养物质与疾病之间的关系。

## 初赛任务

本赛道将提供脱敏后的食物与疾病特征，参赛团队根据主办方提供数据，在高度稀疏数据的场景中，进一步挖掘、融合特征并设计模型，以预测食物与疾病的关系。初赛阶段为二分类问题，分类标签分别为 0（无关）、1（存在正面或负面的影响）。

## 复赛任务

在初赛的基础上，增加对食物与疾病相关性评级维度的评估。
具体的数据说明与评估指标可见比赛链接详细介绍。

# 2. 基本思路

![方案框架.png](https://cdn.nlark.com/yuque/0/2023/png/34514611/1682338826884-59fb2177-9bfd-4f95-ade1-d20fb6f55cf9.png#averageHue=%23070707&clientId=uefa2d5fa-387f-4&from=paste&height=1044&id=u3b06760b&name=%E6%96%B9%E6%A1%88%E6%A1%86%E6%9E%B6.png&originHeight=1566&originWidth=4229&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=242242&status=done&style=none&taskId=u587567e5-60b1-4618-9e14-c6445863378&title=&width=2819.3333333333335)

## 数据预处理

### 读取数据

对于训练数据中的食物特征集（train_food.csv）、疾病特征集（disease_feature1.csv、disease_feature2.csv、disease_feature3.csv）、食物疾病关系数据（train_answer.csv）、食物与疾病关系的相关性评级（semi_train_answer.csv），进行直接读取。然后，将semi_train_answer.csv进行预处理，将related_score（取值为0，1，2，3，4）中取值为0返回0；取值为1，2，3，4返回1，得到"is_related"字段，构成既包含"is_related"字段又包含"related_score"的训练数据（processing_semi_train_answer.csv）。

### 对food_id与disease_id进行编码

通过直接采用food_id与disease_id中的数字对food_id与disease_id进行编码。

### 目标编码

由于数据集中只有两个离散变量food和disease，而测试集中都是新的food，所以用于目标编码的离散字段只有"disease"，本方法采用五折分层抽样的方法对标签"is_related"进行目标编码，得到特征"disease_is_related_mean"。

### 疾病特征数据归一化、降维处理与缺失值填充

由于本方法通过PCA方法对疾病特征数据降维，因此有必要先对疾病特征数据归一化。首先，本方法采用用0填充降维后的疾病特征数据中的缺失值。其次，本方法通过MinMaxScaler()最大最小值的归一化方法对疾病特征数据进行归一化。归一化完毕之后，采用PCA方法对疾病特征数据降维，分别将disease_feature1由996维降至128维，将disease_feature2由3181维降至144维，将disease_feature3由1453维降至256维。

### 数据合并

经过上述数据预处理后，采用merge函数将疾病特征数据与食物特征数据进行合并。

## 特征工程

### 重要特征做log1p变换

通过EDA分析，数据中很多特征存在长尾效应，会影响模型训练和泛化的效果。因此，根据模型输出的特征重要性，本方法采用np.log1p方法对部分重要的特征做log1p变换，去除长尾效应。（截尾处理）
存在长尾效应的特征：N_14、N_59、N_60、N_61、N_85， N_165，N_198， N_193，N_204，N_211（如下图为N_14存在长尾效应的情况）

![image.png](https://cdn.nlark.com/yuque/0/2023/png/34514611/1682338874364-2548a4c6-cb53-4815-8428-4fdef915f209.png#averageHue=%23efeff5&clientId=uefa2d5fa-387f-4&from=paste&height=153&id=ucaab5fb7&name=image.png&originHeight=229&originWidth=349&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=16151&status=done&style=none&taskId=ud12aa083-89cd-4285-82d4-cfdbce1ca39&title=&width=232.66666666666666#pic_center)

```python
# 重要特征做 log1p 变换
cols = ["N_14", "N_59", "N_60", "N_61", "N_85", "N_165", "N_198", "N_193", "N_204", "N_211"]

def log1p(df, col):
    df[col] = np.log1p(df[col])

for c in tqdm(cols):
    log1p(data, c)
```

### 重要特征交叉

特征交叉可以将样本映射至高维空间，从而增加模型的非线性能力,提升模型的预测效果。因此，本方法对部分重要特征（与标签相关性强，具有区分能力的特征），包含"N_33", "N_42", "N_74", "N_106", "N_111", "N_209", "disease", "food"，进行四则运算的特征交叉，提高模型训练和预测效果。

```python
topn = ["N_33", "N_42", "N_74", "N_106", "N_111", "N_209", "disease", "food"]
for i in range(len(topn)):
    for j in range(i + 1, len(topn)):
        data[f"{topn[i]}+{topn[j]}"] = data[topn[i]] + data[topn[j]]
        data[f"{topn[i]}-{topn[j]}"] = data[topn[i]] - data[topn[j]]
        data[f"{topn[i]}*{topn[j]}"] = data[topn[i]] * data[topn[j]]
        data[f"{topn[i]}/{topn[j]}"] = data[topn[i]] / (data[topn[j]] + 1e-5)
```

### 重要食物特征与疾病特征进行特征交叉

为了结合食物与疾病特征，提升模型效果，本方法对重要的食物特征"N_33"与重要的疾病特征"F_82_x", "F_39_x"进行特征交叉。

```python
topn = ["N_33", "F_82_x", "F_39_x"]
for i in range(len(topn)):
    for j in range(i + 1, len(topn)):
        data[f"{topn[i]}+{topn[j]}"] = data[topn[i]] + data[topn[j]]
        data[f"{topn[i]}-{topn[j]}"] = data[topn[i]] - data[topn[j]]
        data[f"{topn[i]}*{topn[j]}"] = data[topn[i]] * data[topn[j]]
        data[f"{topn[i]}/{topn[j]}"] = data[topn[i]] / (data[topn[j]] + 1e-5)
```

### 重要特征分箱处理

本方法通过EDA分析（绘制概率密度图）发现，部分重要特征在不同的取值上食物与疾病是否相关的概率不一样，并且这种关系是非线性的，且具有较强的区分能力。因此本方法考虑对部分重要特征进行分箱处理，分箱离散化之后更能够刻画这种关系，且能够去噪，模型会更稳定，降低过拟合的风险。结果为：三类（0，1，2）分箱的特征：N_33, N_42, N_74, N_111；四类（0，1，2，3）分箱的特征：N_106；10段：N_209和food；14段：disease。
例如特征N_33：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/34514611/1682339295182-0b8aa272-91c8-4b6b-94bb-f131ec744203.png#averageHue=%23f1f0f4&clientId=uefa2d5fa-387f-4&from=paste&height=205&id=uc3c1b389&name=image.png&originHeight=307&originWidth=1066&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=34421&status=done&style=none&taskId=ub41e7f12-7914-42af-bc5c-6c3bc6b14c9&title=&width=710.6666666666666)

数据发现：数据中N_33是一个连续型变量，无法直接统计 related 的概率，故先展示N_33的概率密度，以及每个值下的 0 和 1 的分布比较。不难发现，N_33的值在0~0.125这个部分的0 1 分布区别很大，密度图非交叉区域面积非常大。
数据启示：可视化中的交叉部分说明N_33和related其实并不是严格的线性关系，数据处理中或许考虑把N_33的值分为三段：0-0.125为一段 0.125-1.25 1.25~max

```python
# N_33
def get_N_33_seg(x):
    if x >= 0 and x <= 0.125:
        return 0
    elif x > 0.125 and x <= 1.25:
        return 1
    elif x > 1.25 and x <= np.max(data["N_33"]):
        return 2

data["N33_seg"] = data["N_33"].apply(lambda x: get_N_33_seg(x))
```

同理，对其他特征也是如此处理。对于多段分箱的，本方法采用pd.qcut()函数进行处理，如N_209：

```python
# N_209
data['N_209_qcut'] = pd.qcut(data['N_209'], 10, labels=False, duplicates='drop')
```

### 统计特征构造

本方法通过相关性分析，对前面特征工程构造的特征研究其与标签"is_related"的相关性，从中选择具有较强相关性的特征按"food"分组进行统计特征构造，分别构造“mean”、“std”、“max”、“min”这四种统计特征，从而提升特征的表达能力。

```python
def static_feature(df, features, groups):
    for method in tqdm(['mean', 'std', 'max', 'min']):
        for feature in features:
            for group in groups:
                df[f'{group}_{feature}_{method}'] = df.groupby(group)[feature].transform(method)
    return df
    
dense_feats = ["N_33*disease", 
"N_42+disease", "N_42-disease", "N_42*disease", 
"N_74*disease", 
"N_111+disease", "N_111-disease", "N_111*disease",
"N_33+F_82_x", "N_33+F_39_x"
]
cat_feats = ['food']
data = static_feature(data, dense_feats, cat_feats)
```

## 特征筛选/选择

### 删除特征重要性较低的特征

为了避免不重要特征给模型训练带来负面影响，本方法根据特征重要性，删除特征重要性<100的特征。

### 删除单一取值的特征

单一取值的特征也会给模型训练带来负面影响，因此方法删除了只有单一取值的特征。

### 删除缺失率过高的特征

本方法通过设置阈值为0.95，删除了数据中缺失率大于等于0.95的特征。

## 模型训练

本方法基于xgboost算法，通过10折（因为试验发现，10折的效果比5折要好）交叉验证训练来训练模型（线上训练）。其中的参数通过optuna工具来自动调参，从而得到在当前特征数据下的最优参数：{'max_depth': 8, 'subsample': 0.6538269745359124, 'min_child_weight': 1.3310020580145827, 'eta': 0.055429005058745304, 'gamma': 1.6102668438238998, 'reg_alpha': 0.011767799429858665, 'reg_lambda': 1.790366108891338e-05, 'colsample_bytree': 0.5433494534450877, 'colsample_bylevel': 0.6918888106850691}。模型训练之后可以得到训练集和测试集上的预测概率，分别为 xgb_oof和xgb_pred。

## 概率后处理

为了拟合线上阈值（0.5），本方法将模型对测试集预测的概率加0.36作为最终测试集的预测概率。另外，为了优化NDCG指标，本方法还做了额外的概率后处理操作： 基于以上的特征工程得到训练特征数据，标签为related_score（取值为0，1，2，3，4），然后基于xgboost10折交叉验证训练训练一个多分类模型（名称为“训练”的Notebook，采用训练数据训练模型，特征工程与上述一致，标签为related_score），并将每一折训练的模型以及ntree_limit保存下来，用于对测试集的related_score进行预测，得到每一对food_id和disease_id的相关性评级。然后为每个评级（0，1，2，3，4）进行归一化处理，将评级映射到[0,1]之间，得到调整因子。根据这个调整因子，使用指数函数来拟合概率与评级的关系，得到后处理之后的概率作为最终测试集概率结果，写入提交文件中。

## 总结与思考

- 这个思路比较能提高分数的操作是特征交叉与特征分箱。
- 概率后处理是自己的一些想法，但是提升分数并不明显。
- 我的方案中并没有很好的利用食物与疾病相关性评级数据，例如与其他训练数据结合，我只是将其作为标签训练模型，对预测的概率做后处理。
- 优化NDCG方面，其实NDCG是推荐任务中的一个指标，我并不是做推荐领域的，所以在优化NDCG上的做法也比较普通，且提分不多。
- 需要评估食物与疾病的相关性评级，可以考虑从推荐的角度来完成该赛题，而不是仅仅局限于数据挖掘的思路
- 继续学习！！！




