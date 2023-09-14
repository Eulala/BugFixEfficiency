import csv
import json
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import *
from contrast_sequential_pattern_mining import *
from sequence_cluster import *
from main import *


def draw_violin_plot(data_path, save_path):
    df = pd.read_csv(data_path)
    # print(df)
    f, ax = plt.subplots(figsize=(11, 6))

    sns.violinplot(data=df, palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True)

    # sns.despine(left=True, bottom=True)
    plt.ylabel('length')
    plt.savefig(save_path, dpi=150)


if __name__ == '__main__':
    initialize()

    data_dir = get_global_val('data_dir') + 'sequences/'
    data1 = load_json_list(data_dir + 'issue_sequences_ansible_long.json')
    data2 = load_json_list(data_dir+'issue_sequences_ansible_short.json')

    res = {'short': [], 'long': []}
    for i in data1:
        res['long'].append(len(i['action_sequence']))
    for i in data2:
        res['short'].append(len(i['action_sequence']))
    # res = {'tensorflow': [], 'ansible': []}
    # for i in data:
    #     res['tensorflow'].append(len(i['action_sequence']))
    # data = load_json_list(data_dir + 'issue_sequences_ansible_long.json') + load_json_list(
    #     data_dir + 'issue_sequences_ansible_short.json')
    # for i in data:
    #     res['ansible'].append(len(i['action_sequence']))
    #
    data = []
    for i in res:
        for j in res[i]:
            data.append([i, j])
    df = pd.DataFrame(data, columns=['type', 'length'])
    print(df)

    figure_dir = get_global_val('figure_dir')
    draw_histplot(df, figure_dir+'ansible_issue_sequence_length', 'ansible_issue_sequence_length')
    draw_boxplot(df, figure_dir + 'ansible_issue_sequence_length_box', 'ansible_issue_sequence_length')
