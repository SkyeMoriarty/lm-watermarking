import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

epsilons = [0.1, 0.3, 0.5]


def get_z_distributions(df_s, df_i, epsilon):
    df_s = df_s[df_s["epsilon"] == epsilon]
    df_i = df_i[df_i["epsilon"] == epsilon]
    sns.kdeplot(df_s["replaced z score"], label="baseline")
    sns.kdeplot(df_i["replaced z score"], label="improved")

    plt.xlabel("Z score")
    plt.ylabel("Density")
    plt.title(f"Z score Distribution when epsilon = {epsilon}")
    plt.legend()
    plt.savefig(f"baseline comparison/Z score Distribution under Deletion when epsilon = {epsilon}.png")
    plt.show()


def get_ppl_ratios(df_s, df_i, epsilon):
    df_s = df_s[df_s["epsilon"] == epsilon]
    df_s["ppl ratio"] = df_s["replaced ppl"] / df_s["original ppl"]
    df_i = df_i[df_i["epsilon"] == epsilon]
    df_i["ppl ratio"] = df_i["replaced ppl"] / df_i["original ppl"]
    sns.kdeplot(df_s["ppl ratio"], label="baseline")
    sns.kdeplot(df_i["ppl ratio"], label="improved")

    plt.xlabel("PPL Ratio")
    plt.ylabel("Density")
    plt.title(f"PPL Ratio Distribution when epsilon = {epsilon}")
    plt.legend()
    plt.savefig(f"baseline comparison/PPL Ratio Distribution under Deletion when epsilon = {epsilon}.png")
    plt.show()


if __name__ == "__main__":
    loc_s = "simple ROC/simple_attack_result(with ppl).csv"
    df_s = pd.read_csv(loc_s, encoding='utf-8')
    loc_i = "p_tuned ROC/p_tuned_attack_result(with ppl).csv"
    df_i = pd.read_csv(loc_i, encoding='utf-8')
    for epsilon in epsilons:
        get_z_distributions(df_s, df_i, epsilon)
