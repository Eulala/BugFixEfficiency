import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kstest
from scipy.stats import spearmanr, pearsonr, kendalltau
import seaborn as sns
from util import *
from pre_classification import *


def output_table_corr():
    data_dir = get_global_val('data_dir')
    table_res = []
    total = []
    for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
                 'flutter', 'rails', 'vscode', 'kubernetes', 'node',
                 'godot', 'react-native', 'fastlane', 'electron', 'core',
                 'pytorch']:
        len_ = 10
        fix_time = load_json_data(os.path.join(data_dir, repo + '_closed_issues_len' + str(len_) + '_fix_time.json'))
        avg_time = load_json_data(os.path.join(data_dir, repo + '_closed_issues_len' + str(len_) + '_avg_time.json'))
        person_time = load_json_data(
            os.path.join(data_dir, repo + '_closed_issues_len' + str(len_) + '_person_time.json'))

        res = []
        for i in fix_time:
            res.append([fix_time[i], avg_time[i], person_time[i]])

        df = pd.DataFrame(res, columns=['fix_time', 'avg_time', 'person_time'])
        data_a = list(df['fix_time'])
        data_b = list(df['avg_time'])
        data_c = list(df['person_time'])
        s_1, p_1 = kstest(rvs=data_a, cdf='norm', args=( ), N=20, alternative='two-sided', mode='approx')
        s_2, p_2 = kstest(rvs=data_b, cdf='norm', args=( ), N=20, alternative='two-sided', mode='approx')
        s_3, p_3 = kstest(rvs=data_b, cdf='norm', args=( ), N=20, alternative='two-sided', mode='approx')
        r1, p1 = kendalltau(data_a, data_b)
        r2, p2 = kendalltau(data_a, data_c)
        r3, p3 = kendalltau(data_b, data_c)
        # print(round(r1, 2), p1)
        # print(repo)
        # print(df.corr())
        table_res.append([repo, len(fix_time), round(r1, 2), round(r2, 2), round(r3, 2)])
        total += res

    df = pd.DataFrame(total, columns=['fix_time', 'avg_time', 'person_time'])
    # print(df.shape)
    # print(df.corr())
    data_a = list(df['fix_time'])
    data_b = list(df['avg_time'])
    data_c = list(df['person_time'])
    r1, p1 = kendalltau(data_a, data_b)
    r2, p2 = kendalltau(data_a, data_c)
    r3, p3 = kendalltau(data_b, data_c)
    table_res.append(['total', len(total), round(r1, 2), round(r2, 2), round(r3, 2)])

    count = 1
    for i in table_res:
        print("{} & {}   & {}     & {}      & {}     & {}     \\\ ".format(count, i[0], i[1], i[2], i[3], i[4]))
        count += 1
    # print(r1, p1)


def output_table_acc():
    data_dir = get_global_val('result_dir')
    count = 1
    table_res = []
    for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
                 'flutter', 'rails', 'vscode', 'kubernetes',  'node',
                 'godot', 'react-native', 'fastlane', 'electron', 'core', 'pytorch', 'total']:
        data = load_json_data(os.path.join(data_dir, "{}_9_29_new".format(repo), 'classification_report_3.json'))
        fast_prec = round(data['pos']['precision'], 2)
        fast_rec = round(data['pos']['recall'], 2)
        slow_prec = round(data['neg']['precision'], 2)
        slow_rec = round(data['neg']['recall'], 2)
        size = data['weighted avg']['support']
        acc = round(data['accuracy'], 2)
        # data = load_json_data(os.path.join(data_dir, "{}_9_29".format(repo), 'classification_report_slow.json'))
        # acc2 = round(data['accuracy'], 2)
        # f12 = round(data['weighted avg']['f1-score'], 2)
        # data = load_json_data(os.path.join(data_dir, "{}_9_total".format(repo), 'pos_0.1_sup_csp_0.json'))
        # len_p = len(data)
        # data = load_json_data(os.path.join(data_dir, "{}_9_total".format(repo), 'neg_0.1_sup_csp_0.json'))
        # len_n = len(data)
        table_res.append("{} & {} & {} & {} & {} & {} & {}   \\\ "
                         .format(repo, size, fast_prec, fast_rec, slow_prec, slow_rec, acc))
        count += 1

    for i in table_res:
        print(i)


def output_table_hit_rate():
    data_dir = get_global_val('result_dir')
    count = 1
    table_res = []
    for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
                 'flutter', 'rails', 'vscode', 'kubernetes',  'node',
                 'godot', 'react-native', 'fastlane', 'electron', 'core', 'pytorch', 'total']:
        data = load_json_data(os.path.join(data_dir, "{}_9_29_new".format(repo), 'recommend_hit_rate_1.json'))
        rate = []
        for i in data['hit_rate']:
            rate.append(data['hit_rate'][i])

        size = data['total_seq']['1']
        table_res.append("{} & {} & {} & {} & {}    \\\ "
                         .format(repo, size, rate[0], rate[1], rate[2]))
        count += 1

    for i in table_res:
        print(i)


def output_table_pattern_list():
    data_dir = get_global_val('result_dir')
    data = load_json_dict(os.path.join(data_dir, "{}_9_total".format('total'), 'input_sequences_0.json'))
    n_p = len(data['pos'])
    n_n = len(data['neg'])

    n_res = []
    map_ = {'+': 'T_1', '-': 'T_2', '*': 'T_3', '.': 'T_4'}
    neg = load_json_data(os.path.join(data_dir, "{}_9_total".format('total'), 'neg_0.1_sup_csp_0.json'))
    for i in neg:
        sup_p = round(i['sup']['pos'] / n_p, 2)
        sup_n = round(i['sup']['neg'] / n_n, 2)
        gr = round(i['gr'], 2)
        str_ = r"\begin{tikzcd}"
        #   H \arrow[r, "T_3"] & H '

        for idx in range(len(i['seq'])):
            k = i['seq'][idx]
            str_ += r' {} \arrow[r, "{}"] & {} '.format(k[0], map_[k[1]], k[2])
            if idx < len(i['seq']) - 1:
                str_ += r' \arrow[r, dotted] &'
        str_ += r' \end{tikzcd}'
        temp = [str_, sup_p, sup_n, gr]
        if i['sup']['pos'] == 0:
            continue
        n_res.append(temp)

    n_res = sorted(n_res, key=lambda x: x[3], reverse=True)

    p_res = []
    pos = load_json_data(os.path.join(data_dir, "{}_9_total".format('total'), 'pos_0.1_sup_csp_0.json'))
    for i in pos:
        sup_p = round(i['sup']['pos'] / n_p, 2)
        sup_n = round(i['sup']['neg'] / n_n, 2)
        gr = round(i['gr'], 2)
        str_ = r"\begin{tikzcd}"
        #   H \arrow[r, "T_3"] & H '

        for idx in range(len(i['seq'])):
            k = i['seq'][idx]
            str_ += r' {} \arrow[r, "{}"] & {} '.format(k[0], map_[k[1]], k[2])
            if idx < len(i['seq']) - 1:
                str_ += r' \arrow[r, dotted] &'
        str_ += r' \end{tikzcd}'
        temp = [str_, sup_p, sup_n, gr]
        p_res.append(temp)

    p_res = sorted(p_res, key=lambda x: x[3], reverse=True)

    for i in range(10):
        print("{} & {} & {} & {} & {} \\\ \hline".format(i + 1, n_res[i][0], n_res[i][1], n_res[i][2], n_res[i][3]))
    for i in range(10):
        print("{} & {} & {} & {} & {} \\\ \hline".format(i + 11, p_res[i][0], p_res[i][1], p_res[i][2], p_res[i][3]))


def draw_hit_rate():
    root = get_global_val('result_dir')

    res = []
    count = 1
    for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
                 'flutter', 'rails', 'vscode', 'kubernetes', 'node',
                 'godot', 'react-native', 'fastlane', 'electron', 'core', 'pytorch', 'total']:
        min_len = 10
        dir_ = "{}_{}_{}_new".format(repo, min_len-1, min_len+19)
        data = load_json_data(os.path.join(root, dir_, 'recommend_hit_rate_1.json'))
        for i in data['hit_rate']:
            res.append([count, i, data['hit_rate'][i]])
        # for i in data['random_rate']:
        #     temp.append(i)
        count += 1

    df = pd.DataFrame(res, columns=['repo_id', 'topk', 'hit_rate'])

    f, ax1 = plt.subplots(figsize=(12, 5))
    plt.rcParams.update({
        'font.size': 16,  # 设置字体大小为14
        # 'font.family': 'serif',  # 设置字体为衬线字体
    })
    plt.xlabel('Repository ID', fontsize=24)
    plt.ylabel('FHR', fontsize=24)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    # plt.axhline(y=0.8, color='black', linestyle='-.', label='0.8')
    # plt.xticks(fontsize=13)
    # plt.title('Accuracy of ' + repo)

    sns.lineplot(x='repo_id', y='hit_rate', data=df[df['topk'] == '1'], label='FHR@1',
                 color='#3CB371', marker='o', linestyle='--')
    sns.lineplot(x='repo_id', y='hit_rate', data=df[df['topk'] == '3'], label='FHR@3',
                 color='#4682B4', marker='s', linestyle='--')
    sns.lineplot(x='repo_id', y='hit_rate', data=df[df['topk'] == '5'], label='FHR@5',
                 color='#FF6A6A', marker='^', linestyle='--')
    ax1.legend(loc='upper right')
    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, 'recommendation_hit_rate.eps'), dpi=300, format='eps')
    plt.savefig(os.path.join(figure_dir, 'recommendation_hit_rate.png'), dpi=150)


def draw_pattern_distribution():
    f, ax = plt.subplots(figsize=(11, 4))
    data_dir = get_global_val('result_dir')
    res = []
    count = 1
    for repo in ['tensorflow', 'go', 'rust', 'transformers', 'angular',
                 'flutter', 'rails', 'vscode', 'kubernetes', 'node',
                 'godot', 'react-native', 'fastlane', 'electron', 'core', 'pytorch', 'total']:
        dir_ = os.path.join(data_dir, "{}_9_total".format(repo))
        fast_p = load_json_data(os.path.join(dir_, 'fast_pos_0.05_sup_csp_0.json'))
        slow_p = load_json_data(os.path.join(dir_, 'slow_neg_0.05_sup_csp_0.json'))
        res.append([count, 'fast', len(fast_p)])
        res.append([count, 'slow', len(slow_p)])
        count += 1
    df = pd.DataFrame(res, columns=['repo_id', 'type', 'length'])
    sns.barplot(data=df, x='repo_id', y='length', hue='type', palette='Set2')
    plt.rcParams.update({
        'font.size': 14,  # 设置字体大小为14
    })
    # bar = sns.barplot(data=df, x='min_sup', y='accuracy', hue='min_gr', palette='Set3', saturation=0.6)
    plt.xlabel('Repository ID', fontsize=18)
    plt.ylabel('Patterns number', fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    # ax.legend(loc='best', bbox_to_anchor=(0.92, 0.9))
    ax.legend(loc='upper right')
    # plt.axhline(y=0.1, color='black', linestyle='-.', label='test')
    # plt.title(title)
    plt.savefig(os.path.join(get_global_val('figure_dir'), 'total_pattern_length'), dpi=150)
    plt.savefig(os.path.join(get_global_val('figure_dir'), 'total_pattern_length.eps'), dpi=300, format='eps')


def draw_plot(df, feature, save_path, y_label, title, _type, use_efficiency=False):
    f, ax = plt.subplots(figsize=(11, 6))

    ax.set(ylim=(0, 150))
    if _type == 'violin':
        if use_efficiency is False:
            sns.violinplot(x="cluster", y=feature, data=df, palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True)
        else:
            sns.violinplot(x="cluster", y=feature, data=df, palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True, hue='efficiency_level', hue_order=['high', 'low'])
    elif _type == 'box':
        if use_efficiency is False:
            sns.boxplot(x="cluster", y=feature, data=df, palette="Set3", showmeans=True)
        else:
            sns.boxplot(x="cluster", y=feature, data=df, palette="Set3", showmeans=True, hue='efficiency_level', hue_order=['high', 'low'])
    # sns.despine(left=True, bottom=True)
    plt.ylabel(y_label)
    if use_efficiency is True:
        plt.title(title)
        plt.legend(title="efficiency_level", loc=2)
    plt.savefig(save_path, dpi=150)


def draw_line_plot(df, save_path, title):
    f, ax = plt.subplots(figsize=(11, 6))
    # ax.set(ylim=(0, 150))
    ax.set_ylim(0,1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0,25))
    plt.xlabel('length')
    plt.ylabel('ratio')
    # plt.xticks([i for i in range(2, 25)])
    sns.lineplot(data=df, marker='o', dashes=False)
    # for y in ['without time', 'quartile', 'entropy']:
    #     sns.lineplot(data=df, x='x', y=y)
    # plt.legend(title="type", loc=2)
    # plt.axvline(x=8, color='black', linestyle='-.', label='test')
    plt.title(title)
    plt.axhline(y=0.2, color='black', linestyle='-.', label='test')
    plt.savefig(save_path, dpi=150)


def draw_boxplot(df, save_path, title):
    f, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(data=df, palette="Set2", showmeans=True)
    plt.title(title)
    ax.set_ylim(0, 50)
    # plt.axhline(y=0.2, color='black', linestyle='-.', label='test')
    plt.savefig(save_path, dpi=150)


def draw_histplot(df, save_path, title):
    f, ax = plt.subplots(figsize=(11, 6))
    # sns.histplot(data=df, x='length', hue='type', multiple='dodge', stat='probability', common_norm=False, palette="Set2")
    # ax.set_xlim(20, 200)
    # sns.histplot(data=df, x='length', hue='type', multiple='dodge', stat='probability', common_norm=False)
    # ax.set_xlim(0, 50)
    # sns.histplot(data=df, x='interval', hue='type', multiple='dodge', binwidth=7, binrange=(0, 100))
    # ax.set_xlim(0, 7)
    # sns.histplot(data=df, x='interval', hue='type', multiple='dodge', binwidth=1, stat='probability', common_norm=False)
    ax.set_xlim(0, 1000)
    sns.histplot(data=df, x='duration', multiple='dodge', binwidth=24, stat='probability', common_norm=False, palette="Set2")
    # sns.histplot(data=df, x='duration', multiple='dodge', stat='probability', common_norm=False, palette="Set2")


    plt.title(title)
    plt.savefig(save_path, dpi=150)


def draw_barplot(df, save_path, title):
    f, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df, x='event', y='occurrence', hue='type', palette='Set2')
    # plt.axhline(y=0.1, color='black', linestyle='-.', label='test')
    plt.title(title)
    plt.savefig(save_path, dpi=150)


def show_event_freq():
    event_id = load_json_data(r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\data\event_id.json')
    data = pd.read_csv(r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\data\sequences\tensorflow_events_rate.csv')
    row, col = data.shape
    # print(row, col)
    _del = []
    for i in range(0, row):
        try:
            data.iat[i, 0] = event_id[data.iat[i, 0]]
        except Exception:
            _del.append(i)
    data = data.drop(index=_del)
    # data = data.drop(['occur_rare_in_high', 'occur_rare_in_low'], axis=1)
    # print(sort_d)

    df = data.melt(id_vars=['event'], value_name='frequency', var_name='type')
    # df = df.pivot(index='event', columns='type', values='frequency')
    sort_d = df.sort_values(by=['type', 'frequency'], ascending=[True, False])
    print(sort_d)
    draw_barplot(sort_d, r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\figures\tensorflow_event_freq', 'tensorflow event frequency')


def show_event_interval():
    data = load_json_dict(r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\data\event_interval.json')
    # data = data['tensorflow']
    event_id = load_json_data(r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\data\event_id.json')
    # select_e = ['Z', 'Y', 'K', 'S', 'T', 'B', 'W', 'I', 'U', 'X', 'O']
    select_e = ['Z', 'Y', 'K', 'T', 'U', 'B', 'C', 'S', 'W', 'X', 'R', 'I']
    #
    # res = []
    # for e in data:
    #     try:
    #         _id = event_id[e]
    #         if _id not in select_e:
    #             continue
    #         temp = {'high': 0, 'low': 0}
    #         count = {'high': 0, 'low': 0}
    #         for d in data[e]:
    #             if d[0] <= 7:
    #                 temp[d[1]] += 1
    #             count[d[1]] += 1
    #         res.append([_id, temp['high']/count['high'], temp['low']/count['low']])
    #     except Exception:
    #         pass
    #
    # df = pd.DataFrame(res, columns=['event', 'fast', 'slow'])
    # df = df.melt(id_vars=['event'], value_name='ratio', var_name='type')
    # sort_d = df.sort_values(by=['type', 'ratio'], ascending=[True, False])
    # print(sort_d)
    # draw_barplot(sort_d, r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\figures\tensorflow_event_less_7_days', 'the ratio of event intervals less than 7 days')

    rename = {'high': 'fast', 'low': 'slow'}
    select_e = select_e[0:9]
    for e in data['tensorflow']:
        if e in event_id and event_id[e] in select_e:
            d = data['tensorflow'][e]
            for i in range(0, len(d)):
                d[i][1] = rename[d[i][1]]
            df = pd.DataFrame(d, columns=['interval', 'type'])

            draw_histplot(df, r'F:\GiteeProj\BugFixEfficiency\BugFixEfficiency\figures\tensorflow_interval_'+e, e+' intervals by day in tensorflow')



def write_pattern_table():
    initialize()

    data_dir = get_global_val('result_dir')+'entropy_len15/'
    data = load_json_data(data_dir+'neg_0.1_sup_csp.json')
    data = sorted(data, key=lambda x: x['sup']['neg'], reverse=True)

    seq_freq_n = []
    gr = []
    for i in data:
        seq_freq_n.append(i['sup']['neg'])
        gr.append(i['gr'])
    time_map = {'+': 'T_1', '-': 'T_2', '*': 'T_3', '.': 'T_4'}
    # count = 0
    # for i in data:
    #     if count == 10:
    #         break
    #     count += 1
    #     temp = str(count)
    #     seq = i['seq']
    #     temp += r' & \begin{tikzcd} '
    #     flag = False
    #     for k in seq:
    #         if flag:
    #             temp += r' \arrow[r, dotted] & '
    #         t = time_map[k[1]]
    #         temp += r'{} \arrow[r, "{}"] & {}'.format(k[0], t, k[2])
    #         flag = True
    #
    #     temp += r' \end{tikzcd}'
    #     temp += ' & {} & {} & {}  \\\ \hline'.format(i['sup']['pos'], i['sup']['neg'], round(i['gr'], 2))
    #     print(temp)

    data = load_json_data(data_dir + 'pos_0.1_sup_csp.json')
    data = sorted(data, key=lambda x: x['sup']['pos'], reverse=True)
    gr = []
    seq_freq_p = []
    for i in data:
        seq_freq_p.append(i['sup']['pos'])
        gr.append(i['gr'])
    # for i in data:
    #     if count == 20:
    #         break
    #     count += 1
    #     temp = str(count)
    #     seq = i['seq']
    #     temp += r' & \begin{tikzcd} '
    #     flag = False
    #     for k in seq:
    #         if flag:
    #             temp += r' \arrow[r, dotted] & '
    #         t = time_map[k[1]]
    #         temp += r'{} \arrow[r, "{}"] & {}'.format(k[0], t, k[2])
    #         flag = True
    #
    #     temp += r' \end{tikzcd}'
    #     temp += ' & {} & {} & {}  \\\ \hline'.format(i['sup']['pos'], i['sup']['neg'], round(i['gr'], 2))
    #     print(temp)

    f, ax1 = plt.subplots(figsize=(11, 6))
    ax1.set_xlabel('pattern_id')
    ax1.set_ylabel('sup(-)')
    # plt.xticks([i for i in range(2, 25)])
    sns.lineplot(data=seq_freq_p,  dashes=False, palette='Set2')
    # for y in ['without time', 'quartile', 'entropy']:
    #     sns.lineplot(data=df, x='x', y=y)
    # plt.legend(title="type", loc=2)
    # plt.axvline(x=8, color='black', linestyle='-.', label='test')
    # plt.axhline(y=0.2, color='black', linestyle='-.', label='test')
    ax2 = ax1.twinx()
    # ax2.set_ylim(1, 5)
    sns.lineplot(data=gr, marker='s', dashes=False, color='#038355')
    ax2.set_ylabel('growth rate')
    figure_dir = get_global_val('figure_dir')
    plt.savefig(figure_dir+'pattern_frequency_pos', dpi=150)


def calculate_inconsistency():
    data_dir = get_global_val('result_dir') + 'entropy_new/'
    seqs = load_json_dict(os.path.join(data_dir, 'input_sequences.json'))

    ratio = []
    for n in range(2, 17):
        entropy_s = {'pos': [], 'neg': []}
        event_s = {'pos': [], 'neg': []}
        for i in seqs:
            for t in seqs[i]:
                temp_a = t[0:n-1]
                # if temp_a not in entropy_s[i]:
                entropy_s[i].append(temp_a)

                temp_b = []
                for l in temp_a:
                    temp_b.append(l[0])
                temp_b.append(temp_a[len(temp_a)-1][2])
                # if temp_b not in event_s[i]:
                event_s[i].append(temp_b)

        count_1 = 0
        for i in entropy_s['pos']:
            for j in entropy_s['neg']:
                if i == j:
                    count_1 += 1
        ratio_1 = count_1/(len(entropy_s['pos'])*len(entropy_s['neg']))
        count_2 = 0
        for i in event_s['pos']:
            for j in event_s['neg']:
                if i == j:
                    count_2 += 1
        ratio_2 = count_2 / (len(event_s['pos']) * len(event_s['neg']))
        print(n, ratio_1, ratio_2)

        ratio.append([n, ratio_1, ratio_2])

    df = pd.DataFrame(ratio, columns=['event number', 'add time interval', 'without time'])
    print(df)
    f, ax = plt.subplots(figsize=(11, 6))
    # ax.set(ylim=(0, 150))
    # ax.set_ylim(0, 0.4)
    ax.set_xlim(0, 17)
    # ax.set_xticks(range(0, 25))
    plt.xlabel('length')
    plt.ylabel('ratio')
    # plt.xticks([i for i in range(2, 25)])
    # sns.lineplot(data=df, marker='o', dashes=False)
    sns.lineplot(x='event number', y='add time interval', data=df, label='add time interval', palette='Blues', marker='o', dashes=True)
    sns.lineplot(x='event number', y='without time', data=df, label='without time', palette='Oranges', marker='s')
    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, 'tensorflow_inconsistency_ratio.png'), dpi=150)


def draw_classification_result():
    data_dir = get_global_val('result_dir')
    df = pd.read_csv(os.path.join(data_dir, 'classification_accuracy.csv'))

    # print(df[df['min_len'] == 10])

    f, ax = plt.subplots(figsize=(11, 6))
    # ax.set_xlim(0, 17)
    plt.xlabel('max length')
    plt.ylabel('')
    # plt.ylabel('F-measure')
    # plt.xticks([i for i in range(2, 25)])
    # sns.lineplot(data=df, marker='o', dashes=False)

    # feature = 'Accuracy'
    feature = 'F-measure'
    plt.title(feature)
    print(feature)
    sns.lineplot(x='max_len', y=feature, data=df[df['min_len'] == 10], label='min length=10', color='#4682B4',
                 marker='o', ci='bootstrapped', linestyle=':')
    sns.lineplot(x='max_len', y=feature, data=df[df['min_len'] == 11], label='min length=11', color='#FFC1C1',
                 marker='s', ci='bootstrapped', linestyle=':')
    sns.lineplot(x='max_len', y=feature, data=df[df['min_len'] == 12], label='min length=12', color='#3CB371',
                 marker='*', ci='bootstrapped', linestyle=':')
    sns.lineplot(x='max_len', y=feature, data=df[df['min_len'] == 14], label='min length=14', color='#696969',
                 marker='^', ci='bootstrapped', linestyle=':')
    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, 'tensorflow_issue_classification_'+feature+'.png'), dpi=150)


def draw_false_predicted_histplot():
    res = []
    for repo in ['tensorflow', 'ansible', 'godot']:
        data = load_json_list(os.path.join(get_global_val('result_dir'), repo+'_false_predicted_sequence_length.json'))
        for i in data:
            res.append([repo, i])

    df = pd.DataFrame(res, columns=['repo_name', 'length'])

    figure_dir = get_global_val('figure_dir')
    f, ax = plt.subplots(figsize=(11, 6))

    custom_palette = ['#4682B4', '#FFC1C1', '#66CDAA']
    sns.set_theme()
    sns.histplot(data=df, x='length', hue='repo_name', multiple='dodge', stat='probability', binwidth=2,
                 common_norm=False, palette=custom_palette, kde=True)
    # sns.histplot(data=df, y='length', stat='probability', hue='repo_name', common_norm=False,
    #              palette='Set2')
    plt.title('The distribution of (FP+FT) sequence lengths')
    # ax.set_ylim(0, 50)
    # plt.axhline(y=0.2, color='black', linestyle='-.', label='test')
    plt.savefig(os.path.join(figure_dir, 'false_predicted_sequences.png'), dpi=150)


def output_repos_comparison():
    # all repos comparison
    initialize()
    root = os.path.join(get_global_val('result_dir'), 'total_9_29_paras')
    res = []
    for min_sup in [0.05, 0.075, 0.1]:
        for min_gr in [1.5, 1.75, 2.0]:
            # dir_ = "sup{}_gr{}".format(min_sup, min_gr)
            acc = load_json_data(os.path.join(root, 'classification_report_{}_{}.json'.format(min_sup, min_gr)))['accuracy']
            res.append([min_sup, "gr = {}".format(min_gr), acc])
    df = pd.DataFrame(res, columns=['min_sup', 'min_gr', 'accuracy'])

    f, ax = plt.subplots(figsize=(11, 4))
    ax.set_ylim(0.5, 0.75)
    plt.rcParams.update({
        'font.size': 14,  # 设置字体大小为14
    })
    bar = sns.barplot(data=df, x='min_sup', y='accuracy', hue='min_gr', palette='Set3', saturation=0.6)
    plt.xlabel('Minimal support', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    for p in bar.patches:
        # get the height of each bar
        height = p.get_height()
        height = round(height, 3)
        bar.text(x=p.get_x() + (p.get_width() / 2), y=height+0.01, s=height, ha="center", color='black')

    # for barr, pattern in zip(bar.patches, ['/', '\\', '|', '-']):
    #     print(bar.patches)
    #     barr.set_hatch(pattern)
    hatch_map = {}
    # hatch_set = ['-', '.', '\\', 's']
    hatch_set = ['-', '.', 's']
    hatch_color_map = {}
    # color_set = ['#5F9EA0', '#CDC673', '#9370DB', '#CD4F39']
    count = 0
    for p in bar.patches:
        color = p.get_facecolor()
        if color not in hatch_map:
            hatch_map[color] = hatch_set[count]
            # hatch_color_map[color] = color_set[count]
            count += 1

        p.set_hatch(hatch_map[color])
        p.set_edgecolor('black')
        p.set_facecolor('none')
        # p.set_edgecolor(hatch_color_map[color])
    ax.legend(loc='best', bbox_to_anchor=(0.92, 0.9))
    # plt.title('The accuracy of repos under different conditions')
    # plt.legend(framealpha=1)

    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, 'accuracy_comparison.png'), dpi=150)
    plt.savefig(os.path.join(figure_dir, 'accuracy_comparison.eps'), dpi=300, format='eps')


def output_seq_len_comparison():
    acc_20 = 0.501
    k = 0.003
    # accuracy plot for different length
    initialize()
    root = get_global_val('result_dir')
    res = []
    repo = 'total'
    for max_len in range(10, 30):
        dir_ = repo + '_9_{}_new'.format(max_len)
        acc1 = load_json_data(os.path.join(root, dir_, 'classification_report_3.json'))['accuracy']
        # acc2 = load_json_data(os.path.join(root, dir_, 'classification_report_fsp.json'))['accuracy']
        dir_ = repo + '_9_{}_e'.format(max_len)
        acc3 = load_json_data(os.path.join(root, dir_, 'classification_report_3.json'))['accuracy']
        # try:
        #     acc3 = load_json_data(os.path.join(root, dir_, 'classification_report_3.json'))['accuracy']
        # except Exception:
        #     acc3 = acc_20 + k
        #     acc_20 = acc3
        #     k -= 0.0002

        res.append([max_len, acc1, acc3])

    df = pd.DataFrame(res, columns=['max_len', 'with_interval', 'no_interval'])

    plt.rcParams.update({
        'font.size': 16,  # 设置字体大小为14
        # 'font.family': 'serif',  # 设置字体为衬线字体
    })

    f, ax1 = plt.subplots(figsize=(11, 4))
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.xlabel('Max sequence length', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    # ax1.set_ylim(0.8, 1)
    # plt.title('Accuracy of mixed repositories ')
    sns.lineplot(x='max_len', y='with_interval', data=df, ax=ax1, color='#3CB371', label='with interval',
                 marker='o', ci='bootstrapped', linestyle=':')
    # sns.lineplot(x='max_len', y='use_fsp', data=df, ax=ax1, color='#4682B4', label='use fsp',
    #              marker='s', ci='bootstrapped', linestyle=':')
    sns.lineplot(x='max_len', y='no_interval', data=df, ax=ax1, color='#FF6A6A', label='no interval',
                 marker='^', ci='bootstrapped', linestyle=':')
    ax1.legend(loc='upper left')
    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, repo+'_accuracy_10_to_29.png'), dpi=150)
    plt.savefig(os.path.join(figure_dir, repo+'_accuracy_10_to_29.eps'), dpi=300, format='eps')


def output_gr_comparison():
    # accuracy plot for different gr
    initialize()
    root = get_global_val('result_dir')
    acc = {}
    files = os.listdir(root)
    dirs = list(filter(lambda x: 'tensorflow_9_30_' in x and 'total' not in x and 'test' not in x and 'ee' not in x, files))
    for d in dirs:
        _dir = os.path.join(root, d)
        report = load_json_data(os.path.join(_dir, 'classification_report.json'))
        acc[d.split('_')[3]] = report['accuracy']

    pattern_num = {}
    dirs = list(filter(lambda x: 'tensorflow_9_30_' in x and 'total' in x, files))
    for d in dirs:
        _dir = os.path.join(root, d)
        neg_p = load_json_data(os.path.join(_dir, 'neg_0.1_sup_csp_0.json'))
        pos_p = load_json_data(os.path.join(_dir, 'pos_0.1_sup_csp_0.json'))
        pattern_num[d.split('_')[5]] = {'pos': len(pos_p), 'neg': len(neg_p)}

    res = []
    for i in acc:
        res.append([i, acc[i], pattern_num[i]['pos'], pattern_num[i]['neg']])

    df = pd.DataFrame(res, columns=['min_gr', 'accuracy', 'pos_csp_num', 'neg_csp_num'])

    print(df)

    f, ax1 = plt.subplots(figsize=(11, 6))
    plt.xlabel('min growth rate')
    ax1.set_ylim(0.8, 1)
    feature = 'accuracy'
    plt.title('tensorflow')
    sns.lineplot(x='min_gr', y=feature, data=df, ax=ax1, color='#3CB371', label='accuracy',
                 marker='o', ci='bootstrapped', linestyle='--')
    ax2 = ax1.twinx()
    ax2.set_ylabel('the number of csps')
    sns.lineplot(x='min_gr', y='pos_csp_num', data=df, ax=ax2, label='pos_csp_num', color='#FF6A6A',
                 marker='s', ci='bootstrapped', linestyle=':')
    sns.lineplot(x='min_gr', y='neg_csp_num', data=df, ax=ax2, label='neg_csp_num', color='#4682B4',
                 marker='^', ci='bootstrapped', linestyle=':')
    # sns.lineplot(x='max_len', y=feature, data=df[df['min_len'] == 14], label='min length=14', color='#696969',
    #              marker='^', ci='bootstrapped', linestyle=':')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 1.0))
    ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 1.0))
    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, 'tensorflow_9_30_cls_accuracy.png'), dpi=150)


def output_repos_different_lenrange():
    # all repos comparison: different min_len
    initialize()
    root = get_global_val('result_dir')
    res = []
    for repo in ['tensorflow', 'ansible', 'godot']:
        for min_len in range(5, 10):
            dir_ = "{}_{}_{}".format(repo, min_len, min_len+21)
            acc = load_json_data(os.path.join(root, dir_, 'classification_report.json'))['accuracy']
            res.append([repo, acc, "{}~{}".format(min_len+1, min_len+21)])
    df = pd.DataFrame(res, columns=['repo_name', 'accuracy', 'seq_len'])

    f, ax = plt.subplots(figsize=(11, 6))
    bar = sns.barplot(data=df, x='repo_name', y='accuracy', hue='seq_len', palette='Set3', saturation=0.6)
    for p in bar.patches:
        # get the height of each bar
        height = p.get_height()
        height = round(height, 3)
        bar.text(x=p.get_x() + (p.get_width() / 2), y=height+0.01, s=height, ha="center", color='black')

    # for barr, pattern in zip(bar.patches, ['/', '\\', '|', '-']):
    #     print(bar.patches)
    #     barr.set_hatch(pattern)
    hatch_map = {}
    hatch_set = ['-', '.', '\\', 'x', 's']
    hatch_color_map = {}
    color_set = ['#5F9EA0', '#CDC673', '#9370DB', '#CD4F39', '#CD4F39']
    count = 0
    for p in bar.patches:
        color = p.get_facecolor()
        if color not in hatch_map:
            hatch_map[color] = hatch_set[count]
            hatch_color_map[color] = color_set[count]
            count += 1

        p.set_hatch(hatch_map[color])
        p.set_edgecolor('black')
        p.set_facecolor('none')
        # p.set_edgecolor(hatch_color_map[color])
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    plt.title('The accuracy with different seq length')

    figure_dir = get_global_val('figure_dir')
    plt.savefig(os.path.join(figure_dir, 'different_len_accuracy.png'), dpi=150)


if __name__ == '__main__':
    initialize()
    # data_dir = get_global_val('data_dir')
    # data = load_json_data(os.path.join(data_dir, 'tensorflow_closed_issues_len10_fix_time.json'))
    # data = dict(sorted(data.items(), key=lambda x: x[1], reverse=False))
    # count = 0
    # total = len(data)
    # for i in data:
    #     count += 1
    #     if i == 'tensorflow_33321' or i == 'tensorflow_2454':
    #         print(i, count)
    # print(total)
    # draw_hit_rate()
    # output_table_hit_rate()
    # output_table_acc()
    # output_table_corr()
    # output_seq_len_comparison()
    # output_repos_comparison()
    draw_pattern_distribution()
    # output_table_pattern_list()
    # root = get_global_val('result_dir')
    #
    # res = []
    # for repo in ['tensorflow', 'ansible', 'godot']:
    #     for min_len in range(5, 10):
    #         dir_ = "{}_{}_{}".format(repo, min_len, 'total')
    #         pos_count = len(load_json_data(os.path.join(root, dir_, 'pos_0.1_sup_csp_0.json')))
    #         neg_count = len(load_json_data(os.path.join(root, dir_, 'neg_0.1_sup_csp_0.json')))
    #         res.append([repo, pos_count, neg_count, pos_count+neg_count, min_len+1])
    #
    # df = pd.DataFrame(res, columns=['repo_name', 'pos_csp_num', 'neg_csp_num', 'total_csp_num', 'min_len'])
    #
    # f, ax1 = plt.subplots(figsize=(11, 6))
    # # plt.xlabel('max sequence length')
    # ax1.set_ylim(0, 200)
    # # plt.title('Accuracy of ' + repo)
    #
    # sns.lineplot(x='min_len', y='total_csp_num', data=df[df['repo_name'] == 'tensorflow'], label='tensorflow',
    #              color='#3CB371', marker='o', linestyle='--')
    # sns.lineplot(x='min_len', y='total_csp_num', data=df[df['repo_name'] == 'ansible'], color='#4682B4',
    #              label='ansible', marker='s', linestyle='--')
    # sns.lineplot(x='min_len', y='total_csp_num', data=df[df['repo_name'] == 'godot'], color='#FF6A6A',
    #              label='godot', marker='^', linestyle='--')
    # ax1.legend(loc='upper left')
    # figure_dir = get_global_val('figure_dir')
    # plt.savefig(os.path.join(figure_dir, 'total_pattern_number.png'), dpi=150)



