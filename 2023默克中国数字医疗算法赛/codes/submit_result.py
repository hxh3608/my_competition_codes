# 导入所需要的包
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error, \
    precision_recall_curve, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from sklearn.pipeline import Pipeline

from tqdm import tqdm
# from datetime import datetime
import sys
import os
import gc
# import argparse
import warnings

warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE  # 导入tsne包
from sklearn.decomposition import PCA, KernelPCA  # PCA
from sklearn.manifold import Isomap  # Isomap


def invoke(feature1_path, feature2_path, feature3_path, semi_food_path, dest_path):
    '''
    feature1 feature2 feature3 path：对应三种疾病特征数据路径 已知
    semi_food_path：对应测试集中食物特征的路径 未知
    dest_path：对应结果文件的输出路径
    '''
    # 读取数据
    disease_feature1 = pd.read_csv(feature1_path)['disease_id']
    disease_feature2 = pd.read_csv(feature2_path)['disease_id']
    disease_feature3 = pd.read_csv(feature3_path)['disease_id']
    food_feature = pd.read_csv(semi_food_path)[['food_id']].assign(key=1)  # 141636

    # 创建提交文件
    all_disease = pd.DataFrame(
        {"disease_id": list(set(disease_feature1).union(set(disease_feature2)).union(set(disease_feature3)))}).assign(
        key=1)  # 407
    result_df = food_feature.merge(all_disease, how='outer', on='key').drop(columns=['key'])

    def read_and_process_data():
        # 读取疾病特征数据集（训练集）
        disease_feature1 = pd.read_csv(feature1_path)
        disease_feature2 = pd.read_csv(feature2_path)
        disease_feature3 = pd.read_csv(feature3_path)

        # 读取 train_food 和 train_answer（训练集）
        train_food = pd.read_csv("./data/train_food.csv")
        train_answer = pd.read_csv("./data/processing_semi_train_answer.csv")

        # 读取测试集食物数据
        test_food = pd.read_csv(semi_food_path)  # 测试集食物特征

        # 自己根据 submit_result 构建 submit
        all_disease = pd.DataFrame({"disease_id": list(
            set(disease_feature1['disease_id']).union(set(disease_feature2['disease_id'])).union(
                set(disease_feature3['disease_id'])))}).assign(key=1)
        food_feature = test_food[['food_id']].assign(key=1)
        submit = food_feature.merge(all_disease, how='outer', on='key').drop(columns=['key'])

        # 拼接标签数据（将包含标签的训练集和没有标签的测试集拼接）
        data = pd.concat([train_answer, submit], axis=0).reset_index(drop=True)

        # 拼接食物数据（将训练集食物数据和测试集食物数据拼接）
        food = pd.concat([train_food, test_food], axis=0).reset_index(drop=True)

        # 对food_id和disease_id进行编码
        data["food"] = data["food_id"].apply(lambda x: int(x.split("_")[-1]))
        data["disease"] = data["disease_id"].apply(lambda x: int(x.split("_")[-1]))

        # 目标编码
        cat_list = ['disease']

        def stat(df, df_merge, group_by, agg):
            group = df.groupby(group_by).agg(agg)

            columns = []
            for on, methods in agg.items():
                for method in methods:
                    columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
            group.columns = columns
            group.reset_index(inplace=True)
            df_merge = df_merge.merge(group, on=group_by, how='left')

            del (group)
            gc.collect()
            return df_merge

        def statis_feat(df_know, df_unknow, cat_list):  # 统计特征
            for f in tqdm(cat_list):
                df_unknow = stat(df_know, df_unknow, [f], {'is_related': ['mean']})

            return df_unknow

        df_train = data[~data['is_related'].isnull()]
        df_train = df_train.reset_index(drop=True)
        df_test = data[data['is_related'].isnull()]

        df_stas_feat = None
        kf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
        for train_index, val_index in kf.split(df_train, df_train['is_related']):
            df_fold_train = df_train.iloc[train_index]
            df_fold_val = df_train.iloc[val_index]

            df_fold_val = statis_feat(df_fold_train, df_fold_val, cat_list)
            df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

            del (df_fold_train)
            del (df_fold_val)
            gc.collect()

        df_test = statis_feat(df_train, df_test, cat_list)
        data = pd.concat([df_stas_feat, df_test], axis=0)
        data = data.reset_index(drop=True)

        del (df_stas_feat)
        del (df_train)
        del (df_test)

        return disease_feature1, disease_feature2, disease_feature3, data, food

    # 降维前归一化
    def standard_disease(df):
        """
        降维前归一化
        """
        std = MinMaxScaler()
        disease_id_array = df["disease_id"]
        cols = [f for f in df.columns if f not in ["disease_id"]]
        df_std = std.fit_transform(df[cols])
        df_temp = pd.DataFrame(data=df_std[0:, 0:], columns=cols)
        df_disease_id = pd.DataFrame(data=disease_id_array, columns=["disease_id"])
        df = pd.concat([df_disease_id, df_temp], axis=1)
        return df

    # PCA降维
    ## PCA 对疾病特征数据集进行降维处理
    def pca(df, n):
        disease_id_array = df["disease_id"]
        df_pca = PCA(n_components=n).fit_transform(df.iloc[:, 1:])
        df_temp = pd.DataFrame(data=df_pca[0:, 0:], columns=["F_" + str(item) for item in range(df_pca.shape[1])])
        df_disease_id = pd.DataFrame(data=disease_id_array, columns=["disease_id"])
        df_disease = pd.concat([df_disease_id, df_temp], axis=1)
        print(df_disease.shape)
        return df_disease

    def merge():
        disease_feature1, disease_feature2, disease_feature3, data, food = read_and_process_data()

        disease_feature1 = disease_feature1.fillna(0)
        disease_feature2 = disease_feature2.fillna(0)
        disease_feature3 = disease_feature3.fillna(0)

        # 归一化
        disease_feature1 = standard_disease(disease_feature1)
        disease_feature2 = standard_disease(disease_feature2)
        disease_feature3 = standard_disease(disease_feature3)

        # pca
        df_disease1 = pca(disease_feature1, 128)
        df_disease2 = pca(disease_feature2, 144)
        df_disease3 = pca(disease_feature3, 256)

        # 数据合并
        data = pd.merge(data, food, on="food_id", how="left")
        data = pd.merge(data, df_disease1, on="disease_id", how="left")
        data = pd.merge(data, df_disease2, on="disease_id", how="left")
        data = pd.merge(data, df_disease3, on="disease_id", how="left")

        # 重要特征做 log1p 变换
        cols = ["N_14", "N_59", "N_60", "N_61", "N_85", "N_165", "N_198", "N_193", "N_204", "N_211"]

        def log1p(df, col):
            df[col] = np.log1p(df[col])

        for c in tqdm(cols):
            log1p(data, c)

        return data

    def feature_engineering():
        data = merge()

        # 特征工程
        ## 特征交叉
        topn = ["N_33", "N_42", "N_74", "N_106", "N_111", "N_209", "disease", "food"]
        for i in range(len(topn)):
            for j in range(i + 1, len(topn)):
                data[f"{topn[i]}+{topn[j]}"] = data[topn[i]] + data[topn[j]]
                data[f"{topn[i]}-{topn[j]}"] = data[topn[i]] - data[topn[j]]
                data[f"{topn[i]}*{topn[j]}"] = data[topn[i]] * data[topn[j]]
                data[f"{topn[i]}/{topn[j]}"] = data[topn[i]] / (data[topn[j]] + 1e-5)
        
        ## 特征交叉 重要食物特征与疾病特征
        topn = ["N_33", "F_82_x", "F_39_x"]
        for i in range(len(topn)):
            for j in range(i + 1, len(topn)):
                data[f"{topn[i]}+{topn[j]}"] = data[topn[i]] + data[topn[j]]
                data[f"{topn[i]}-{topn[j]}"] = data[topn[i]] - data[topn[j]]
                data[f"{topn[i]}*{topn[j]}"] = data[topn[i]] * data[topn[j]]
                data[f"{topn[i]}/{topn[j]}"] = data[topn[i]] / (data[topn[j]] + 1e-5)

        ## 重要特征分箱处理
        ### N_33
        def get_N_33_seg(x):
            if x >= 0 and x <= 0.125:
                return 0
            elif x > 0.125 and x <= 1.25:
                return 1
            elif x > 1.25 and x <= np.max(data["N_33"]):
                return 2

        data["N33_seg"] = data["N_33"].apply(lambda x: get_N_33_seg(x))

        ### N_42
        def get_N_42_seg(x):
            if x >= 0 and x <= 500:
                return 0
            elif x > 500 and x <= 1000:
                return 1
            elif x > 1000 and x <= np.max(data["N_42"]):
                return 2

        data["N42_seg"] = data["N_42"].apply(lambda x: get_N_42_seg(x))

        ### N_74
        def get_N_74_seg(x):
            if x >=0 and x <= 2:
                return 0
            elif x > 2 and x <= 10:
                return 1
            elif x > 10 and x <= np.max(data["N_74"]):
                return 2

        data["N74_seg"] = data["N_74"].apply(lambda x: get_N_74_seg(x))

        ### N_106
        def get_N_106_seg(x):
            if x >= 0 and x <= 50:
                return 0
            elif x > 50 and x <= 150:
                return 1
            elif x > 150 and x <= 300:
                return 2
            elif x > 300 and x < np.max(data["N_106"]):
                return 3

        data["N106_seg"] = data["N_106"].apply(lambda x: get_N_106_seg(x))

        ### N_111
        def get_N_111_seg(x):
            if x >= 0 and x <= 500:
                return 0
            elif x > 500 and x <= 2000:
                return 1
            elif x > 2000 and x <= np.max(data["N_111"]):
                return 2

        data["N111_seg"] = data["N_111"].apply(lambda x: get_N_111_seg(x))

        ### food
        data['food_qcut'] = pd.qcut(data['food'], 10, labels=False, duplicates='drop')

        ### disease
        data['disease_qcut'] = pd.qcut(data['disease'], 14, labels=False, duplicates='drop')

        ### N_209
        data['N_209_qcut'] = pd.qcut(data['N_209'], 10, labels=False, duplicates='drop')

        def static_feature(df, features, groups):
            for method in tqdm(['mean', 'std', 'max', 'min']):
                for feature in features:
                    for group in groups:
                        df[f'{group}_{feature}_{method}'] = df.groupby(group)[feature].transform(method)
            return df
                
        ### 有0.0005的分数提升 
        dense_feats = ["N_33*disease", 
        "N_42+disease", "N_42-disease", "N_42*disease", 
        "N_74*disease", 
        "N_111+disease", "N_111-disease", "N_111*disease",
        "N_33+F_82_x", "N_33+F_39_x"
        ]
        cat_feats = ['food']
        data = static_feature(data, dense_feats, cat_feats)

        # 根据特征重要性筛选特征
        feat_imp = pd.read_csv("/home/mw/project/files/features_importance.csv")
        feat_no_imp = feat_imp[(feat_imp["imp"] < 100)].reset_index(drop=True)
        no_imp_cols = feat_no_imp.feats.tolist()
        data = data.drop(no_imp_cols, axis=1)

        return data

    # 特征筛选/处理
    def feature_processing():

        data = feature_engineering()

        drop_cols = ["disease_id", "food_id", "is_related", "related_score"]

        ## 去除只有单一取值的特征
        for col in data.columns:
            if data[col].nunique() < 2:
                drop_cols.append(col)

        cols = pickle.load(open("/home/mw/project/files/drop_cols", "rb"))
        data = data.drop(cols, axis=1)

        return data, drop_cols

    # 划分 train 和 test
    def spilt_train_test():
        data, drop_cols = feature_processing()

        df_test = data[data["is_related"].isnull() == True].copy().reset_index(drop=True)
        df_train = data[~data["is_related"].isnull() == True].copy().reset_index(drop=True)

        features_name = [f for f in df_train.columns if f not in drop_cols]
        

        x_train = df_train[features_name].reset_index(drop=True)
        x_test = df_test[features_name].reset_index(drop=True)
        y = df_train["is_related"].reset_index(drop=True)

        return df_test, x_train, x_test, y

    def cv_model(clf, train_x, train_y, test_x, clf_name):
        folds = 10
        seed = 2023
        # kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        oof = np.zeros(train_x.shape[0])  # 初始化一个大小为n（n=训练集行数），值全为0的数组 用于存放每折验证集的预测概率
        predict = np.zeros(test_x.shape[0])  # 初始化一个大小为n（n=测试集行数），值全为0的数组 用于存放预测概率

        cv_scores = []

        for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
            print('************************************ {} ************************************'.format(str(i + 1)))
            trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
                                         train_y[valid_index]

            if clf_name == "lgb":
                train_matrix = clf.Dataset(trn_x, label=trn_y)  # 该折训练集矩阵
                valid_matrix = clf.Dataset(val_x, label=val_y)  # 该折验证集矩阵

                params = {
                    'learning_rate': 0.01,
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'num_leaves': 63,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'seed': 2022,
                    'bagging_seed': 1,
                    'feature_fraction_seed': 7,
                    'min_data_in_leaf': 20,
                    'verbose': -1,
                    'n_jobs': 8
                    # 'device':'gpu',
                    # 'gpu_platform_id':0,
                    # 'gpu_device_id':0

                }
                # 模型训练 valid_sets也可以只放valid_matrix verbose_eval表示打印信息的间隔 early_stopping_rounds表示早停，
                # 防止过拟合，表示在验证集上,当连续n次迭代,分数没有提高后,提前终止训练
                model = clf.train(params, train_matrix, 100000, valid_sets=[train_matrix, valid_matrix],
                                  categorical_feature=[], verbose_eval=100, early_stopping_rounds=500)
                val_pred = model.predict(val_x, num_iteration=model.best_iteration)  # 预测该折验证集 最优迭代次数
                test_pred = model.predict(test_x, num_iteration=model.best_iteration)  # 该折训练下的模型来预测测试集

                print(list(
                    sorted(zip(features_name, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[
                      :30])

            if clf_name == "xgb":
                train_matrix = clf.DMatrix(trn_x, label=trn_y)  # 该折该折训练集矩阵
                valid_matrix = clf.DMatrix(val_x, label=val_y)  # 该折验证集矩阵
                test_matrix = clf.DMatrix(test_x)  # 测试集矩阵

                params = {'booster': 'gbtree',
                          'objective': 'binary:logistic',
                          'eval_metric': 'auc',
                          'gamma': 1.6102,
                          'min_child_weight': 1.331,
                          'max_depth': 8,
                          'subsample': 0.6538,
                          'colsample_bytree': 0.5433,
                          'colsample_bylevel': 0.7,
                          'reg_alpha':0.0118,
                          'reg_lambda':1.79e-05,
                          'eta': 0.0554,
                          'seed': 2020,
                          'nthread': 8,
                          'gpu_id': 0,
                          'tree_method': 'gpu_hist'
                          }

                watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]
                # num_boost_round为迭代次数 evals是一个列表，用于对训练过程中进行评估列表中的元素，形式就是watchlist
                model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=100,
                                  early_stopping_rounds=500)
                val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)  # 最优模型时对应树的个数
                test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)

            if clf_name == "cat":
                model = clf(
                    n_estimators=10000,
                    random_seed=1024,
                    eval_metric='AUC',
                    learning_rate=0.05,
                    max_depth=5,
                    early_stopping_rounds=500,
                    metric_period=500,
                    task_type='GPU'
                )

                model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                          use_best_model=True,
                          verbose=1)

                val_pred = model.predict_proba(val_x)[:, 1]
                test_pred = model.predict_proba(test_x)[:, 1]

            oof[valid_index] = val_pred  # 将每一折验证集的预测结果放入原先的初始化矩阵中（每一折会对应索引valid_index）

            predict += test_pred / folds

            cv_scores.append(roc_auc_score(val_y, val_pred))
            print(cv_scores)

        return oof, predict
    
    def get_related_score():
        data = feature_engineering()
        df_test = data[data["is_related"].isnull() == True].copy().reset_index(drop=True)

        features_name = pickle.load(open("/home/mw/project/files/features_name", "rb"))

        features_name = sorted(features_name)

        x_test = df_test[features_name].reset_index(drop=True)

        xgb_test_matrix = xgb.DMatrix(x_test)
        
        xgb_predict = np.zeros((x_test.shape[0], 5))

        for i in range(10):
            xgb_model = pickle.load(open(f"/home/mw/project/model/xgb_multiclass_0/xgb_model{i}", "rb"))
            xgb_model_best_limit = pickle.load(open(f"/home/mw/project/model/xgb_multi_best_ntree_limit_0/xgb_model_best_ntree_limit{i}", "rb"))
            xgb_test_pred = xgb_model.predict(xgb_test_matrix, ntree_limit=xgb_model_best_limit)
            xgb_predict += xgb_test_pred / 10
        
        return xgb_predict

    def get_submit():
        # 模型训练 二分类

        df_test, x_train, x_test, y = spilt_train_test()

        xgb_oof, xgb_pred = cv_model(xgb, x_train, y, x_test, 'xgb')

        df_test['related_prob'] = xgb_pred + 0.36

        result = df_test['related_prob']

        return result

    result = get_submit()
    result_df['related_prob'] = result

    xgb_predict = get_related_score()
    result_df["related_score"] = np.argmax(xgb_predict, axis=1)

    ## 概率后处理
    # 根据评级对预测结果进行后处理，调整预测概率
    related_score = result_df["related_score"]
    related_prob = result_df['related_prob']
    related_score_norm = related_score / max(related_score) # 对评级进行归一化处理，将评级映射到[0,1]之间
    adjust_factor = np.exp(related_score_norm) # 根据评级得到调整因子，使用指数函数来拟合调整函数关系
    prob_adjusted = related_prob * adjust_factor # 对预测概率进行调整

    result_df = result_df.drop(["related_score", "related_prob"], axis=1)

    result_df["related_prob"] = prob_adjusted

    # 写出提交文件
    result_df.to_csv(dest_path, index=False)



