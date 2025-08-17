import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

epsilons = [0, 0.1, 0.3, 0.5]
types = ['original', 'replaced', 'inserted', 'deleted']

model_name = "facebook/opt-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # 推理模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def filter_epsilon(df, epsilon):
    return df[df['epsilon'] == epsilon]


def count_TP(df, column_name):
    return (df[column_name] == 'Watermarked').sum()


def calculate_mean_z(df, column_name):
    return df[column_name].mean()


# 此df为某种attack下的filtered df
# def get_error_rate_csv(df, z_thresholds, type):
#     res = pd.DataFrame(columns=['Epsilon', 'count', 'TPR', 'FNR',
#                                 'TPR with attack', 'FNR with attack'])
#
#     for epsilon in epsilons:
#         filtered_df = filter_epsilon(df, epsilon)
#         count = len(filtered_df)
#         row = [epsilon, count]
#         TP = np.where(filtered_df, filtered_df[type + ' z threshold']>)
#         FN = count - TP
#         row.append(TP / count)
#         row.append(FN / count)
#     print(row)
#     res = pd.concat([
#         res,
#         pd.DataFrame([row], columns=res.columns)
#     ], ignore_index=True)
#
#     res.to_csv('test.csv')


# z值的mean&std、mean green ratio、TPR&FNR @ a fixed z
# 这里用到的只有original watermarked completion
def get_comparison_metrics(df, z_thresholds):
    filtered_df = filter_epsilon(df, 0.1)
    count = len(filtered_df)
    z_scores = filtered_df['original z score']
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    green_fractions = filtered_df['original green fraction']
    decimal_list = [float(x.replace("%", "")) / 100 for x in green_fractions]
    green_ratio_mean = np.mean(decimal_list)
    TPRs = []
    FNRs = []
    for threshold in z_thresholds:
        TP_count = (filtered_df['original z score'] >= threshold).sum()
        TPRs.append(TP_count / count)
        FNRs.append((count - TP_count) / count)

    return z_mean, z_std, green_ratio_mean, TPRs, FNRs


def find_best_threshold(z_watermark, z_baseline):
    thresholds = np.linspace(0, 10, 200)
    best_threshold = None

    for t in thresholds:
        tpr = (z_watermark >= t).mean()
        fpr = (z_baseline >= t).mean()
        if tpr >= 0.9 and fpr <= 0.05:
            best_threshold = t
            break

    return best_threshold


def calculate_ppl(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # 计算loss（shifted）
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    # perplexity = e^loss
    perplexity = math.exp(loss.item())
    return perplexity


def calculate_ppls(texts):
    ppls = []
    for text in texts:
        ppls.append(calculate_ppl(text))
    return ppls


def save_best_thresholds(thresholds, dir):
    df = pd.DataFrame(thresholds, index=types[1:], columns=epsilons)
    df.to_csv("ablation study/" + dir.split()[0] + " best thresholds.csv")


def get_ROC(df, dir):
    best_thresholds = [[0 for _ in range(4)] for _ in range(3)]
    print(np.shape(best_thresholds))
    i = 0
    for type in types[1:]:
        plt.figure(figsize=(8, 6))
        j = 0
        for epsilon in epsilons:
            if epsilon == 0:
                filtered_df = filter_epsilon(df, 0.1)
                count = len(filtered_df)
                y_watermark = filtered_df['original z score']
                texts = filtered_df['original watermarked completion']
            else:
                filtered_df = filter_epsilon(df, epsilon)
                count = len(filtered_df)
                y_watermark = filtered_df[type + ' z score']
                texts = filtered_df[type + ' watermarked completion']

            y_baseline = filtered_df['baseline z score']
            y_predict = list(y_watermark) + list(y_baseline)

            y_true = [1] * count + [0] * count
            fpr, tpr, thresholds = roc_curve(y_true, y_predict)
            roc_auc = auc(fpr, tpr)

    #         # 假设希望 TPR >= 90%, FPR <= 5%
    #         mask = (tpr >= 0.9) & (fpr <= 0.05)
    #         if mask.any():
    #             candidate_thresholds = thresholds[mask]
    #             best_threshold = candidate_thresholds[np.argmin(fpr[mask])]
    #         else:
    #             # fallback: youden's J
    #             youden_index = np.argmax(tpr - fpr)
    #             best_threshold = thresholds[youden_index]
    #         best_thresholds[i][j] = best_threshold
    #         print(i, j, best_threshold)
    #         j += 1
    #     i += 1
    # save_best_thresholds(best_thresholds, dir)

            # 绘制 ROC 曲线
            if epsilon == 0:
                plt.plot(fpr, tpr, label=f'unattacked, AUC = {roc_auc:.3f}')
            else:
                plt.plot(fpr, tpr, label=f'ε = {epsilon}, AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve - {type}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir + f'/ROC Curve - {type}')
        plt.show()


def draw_z_distribution(df_g, df_gp, df_gpa):
    sns.kdeplot(df_g['original z score'], label='G only', linewidth=2)
    sns.kdeplot(df_gp['original z score'], label='G + P', linewidth=2)
    sns.kdeplot(df_gpa['original z score'], label='G + P + A', linewidth=2)

    plt.axvline(x=4, color='gray', linestyle='--', label='z = 4 threshold')
    plt.xlabel('Z Score')
    plt.ylabel('Density')
    plt.title('Z-score Distribution under Different Module Configurations')
    plt.legend()
    plt.grid(True)
    plt.savefig('ablation study/z score distribution')
    plt.show()


# 对比每个模块的贡献度
def get_metrics_comparison(locs, z_thresholds, des):
    columns = {'seeding type': [],
               'z-score mean': [],
               'z-score std': [],
               'green fraction mean': []}
    for z_threshold in z_thresholds:
        columns.update({'TPR@' + str(z_threshold): []})
        columns.update({'FNR@' + str(z_threshold): []})

    for loc in locs:
        columns['seeding type'].append(loc.split()[0])
        df = pd.read_csv(loc, encoding='utf-8')
        z_mean, z_std, green_ratio_mean, TPRs, FNRs = get_comparison_metrics(df, z_thresholds)
        columns['z-score mean'].append(f"{z_mean:.2f}")
        columns['z-score std'].append(f"{z_std:.2f}")
        columns['green fraction mean'].append(f"{green_ratio_mean:.2f}")
        for i, (tpr, fnr) in enumerate(zip(TPRs, FNRs)):
            columns['TPR@' + str(z_thresholds[i])].append(f"{tpr:.2f}")
            columns['FNR@' + str(z_thresholds[i])].append(f"{fnr:.2f}")

    res = pd.DataFrame(columns)
    res.to_csv(des, index=False, encoding='utf-8')


if __name__ == '__main__':
    locs = ['g ROC/g_attack_result.csv',
            'g+p ROC/g+p_attack_result.csv',
            'g+p+a ROC/g+p+a_attack_result.csv',
            'p tuned ROC/p_tuned_attack_result.csv']
    #
    # z_thresholds = [3, 4, 5]
    # des = 'ablation study/metrics comparison.csv'
    # get_metrics_comparison(locs, z_thresholds, des)

    df = pd.read_csv(locs[3], encoding='utf-8')
    # dir = locs[3].split('/')[0]
    # get_ROC(df, dir)
    #
    for type in types:
        texts = df[type + ' watermarked completion']
        df[type + ' ppl'] = calculate_ppls(texts)

    df.to_csv('p tuned ROC/p_tuned_attack_result(with ppl).csv')

    # for loc in locs:
    #     dir = loc.split("/")[0]
    #     df = pd.read_csv(loc, encoding='utf-8')
    #     get_ROC(df, dir)
