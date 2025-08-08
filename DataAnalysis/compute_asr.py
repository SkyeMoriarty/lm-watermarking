import pandas as pd

epsilons = [0.1, 0.3, 0.5]
types = ['replaced', 'inserted', 'deleted']


def compute_asr(df, z_threshold, r_max=4):
    results = {}
    for attack_type in types:
        results[attack_type] = {}
        for epsilon in epsilons:
            filtered_df = df[df['epsilon'] == epsilon]
            z_scores = filtered_df[attack_type + ' z score']
            original_ppls = filtered_df['original ppl']
            attacked_ppls = filtered_df[attack_type + ' ppl']

            # 成功攻击的布尔数组
            success_mask = (z_scores < z_threshold) & (attacked_ppls / original_ppls < r_max)

            num_success = success_mask.sum()
            total = len(filtered_df)
            asr = num_success / total if total > 0 else 0

            results[attack_type][epsilon] = {
                "num_success": num_success,
                "total": total,
                "asr": asr
            }
    return results


if __name__ == '__main__':
    loc = 'simple ROC/simple_attack_result(with ppl).csv'
    df = pd.read_csv(loc, encoding='utf-8')
    results = compute_asr(df, 4)

    loc_hs = 'hashed simple ROC/hashed_simple_attack_result(with ppl).csv'
    df = pd.read_csv(loc_hs, encoding='utf-8')
    results_hs = compute_asr(df, 4)

    loc_gpa = 'g+p+a ROC/g+p+a_attack_result(with ppl).csv'
    df_gpa = pd.read_csv(loc_gpa, encoding='utf-8')
    results_gpa = compute_asr(df_gpa, 4)

    rows = []
    for type in types:
        for epsilon in results[type]:
            row = {
                "attack_type": type,
                "epsilon": epsilon,
                "baseline": f"{results[type][epsilon]['asr']:.2%}",
                "hashed baseline": f"{results_hs[type][epsilon]['asr']:.2%}",
                "optimized": f"{results_gpa[type][epsilon]['asr']:.2%}"
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("baseline comparison/asr_comparison.csv", index=False)
