import matplotlib.pyplot as plt
import seaborn as sns
from util import *
from pre_classification import *


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

    sns.boxplot(x="type", y='length', data=df, palette="Set2", showmeans=True)
    plt.title(title)
    ax.set_ylim(0, 200)
    # plt.axhline(y=0.2, color='black', linestyle='-.', label='test')
    plt.savefig(save_path, dpi=150)


def draw_histplot(df, save_path, title):
    f, ax = plt.subplots(figsize=(11, 6))
    # sns.histplot(data=df, x='length', hue='type', multiple='dodge')
    # sns.histplot(data=df, x='length', hue='type', multiple='dodge', stat='probability', common_norm=False)
    # ax.set_xlim(0, 50)
    # sns.histplot(data=df, x='interval', hue='type', multiple='dodge', binwidth=7, binrange=(0, 100))
    # ax.set_xlim(0, 7)
    # sns.histplot(data=df, x='interval', hue='type', multiple='dodge', binwidth=1, stat='probability', common_norm=False)
    ax.set_xlim(20, 100)
    sns.histplot(data=df, x='length', hue='type', multiple='dodge', stat='probability', common_norm=False, palette="Set2")

    plt.title(title)
    plt.savefig(save_path, dpi=150)


def draw_barplot(df, save_path, title):
    f, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df, x='event', y='ratio', hue='type', palette='Set2')
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


def reset_cluster_name(cluster_path, write_path):
    # sort from smallest to largest
    cluster_features = load_json_list(cluster_path)[0]
    loc = { }
    for repo in cluster_features:
        loc[repo] = {}
        for cluster in cluster_features[repo]:
            loc[repo][cluster] = []
            for i in cluster_features[repo][cluster]:
                loc[repo][cluster].append(cluster_features[repo][cluster][i]['loc'])

    loc_medians = {}
    for repo in loc:
        loc_medians[repo] = []
        for cluster in loc[repo]:
            loc_medians[repo].append(numpy.median(loc[repo][cluster]))

    indexes = {}
    for repo in loc_medians:
        print(loc_medians[repo])
        indexes[repo] = numpy.argsort(numpy.array(loc_medians[repo]))

    print(indexes)
    res = {}
    for repo in cluster_features:
        res[repo] = {}
        for i in range(len(indexes[repo])):
            print(i)
            res[repo][i] = cluster_features[repo][str(indexes[repo][i])]
    write_json_list([res], write_path)


def translate_to_csv(read_path, write_path):
    cluster_features = load_json_list(read_path)[0]
    # efficiency = { }
    # for repo in cluster_features:
    #     efficiency[repo] = {}
    #     for cluster in cluster_features[repo]:
    #         efficiency[repo][cluster] = []
    #         for i in cluster_features[repo][cluster]:
    #             efficiency[repo][cluster].append(cluster_features[repo][cluster][i]['fix_efficiency'])
    #
    # efficiency_cut_off = {}
    # for repo in efficiency:
    #     efficiency_cut_off[repo] = {}
    #     for cluster in efficiency[repo]:
    #         efficiency_cut_off[repo][cluster] = numpy.mean(efficiency[repo][cluster])

    res = []
    for repo in cluster_features:
        for cluster in cluster_features[repo]:
            for i in cluster_features[repo][cluster]:
                efficiency_level = 'unsure'
                # if cluster_features[repo][cluster][i]['fix_efficiency'] > efficiency_cut_off[repo][cluster]:
                #     efficiency_level = 'high'
                res.append([repo, i, cluster, cluster_features[repo][cluster][i]['fix_efficiency'], cluster_features[repo][cluster][i]['fix_time'], cluster_features[repo][cluster][i]['sequence_len'], cluster_features[repo][cluster][i]['loc'], efficiency_level])

    with open(write_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['repo_name', 'number', 'cluster', 'fix_efficiency', 'fix_time', 'sequence_len', 'loc', 'efficiency_level'])
        for r in res:
            writer.writerow(r)


def generate_all_pics(read_path):
    df = pd.read_csv(read_path)

    features = ['sequence_len', 'fix_time', 'fix_efficiency']
    # features = ['fix_efficiency']
    for repo in ['tensorflow', 'ansible']:
        temp_df = df[df['repo_name'].isin([repo])]
        for feature in features:
            draw_plot(temp_df, feature, 'figures/' + repo + '_' + feature + '_violin.png', y_label=feature, title=repo,
                      _type='violin')
            draw_plot(temp_df, feature, 'figures/' + repo + '_' + feature + '_box.png', y_label=feature, title=repo,
                      _type='box')

    features = ['sequence_len', 'fix_time']
    for repo in ['tensorflow', 'ansible']:
        temp_df = df[df['repo_name'].isin([repo])]
        for feature in features:
            draw_plot(temp_df, feature, 'figures/' + repo + '_' + feature + '_by_efficiency_violin.png', y_label=feature, title=repo,
                      _type='violin', use_efficiency=True)
            draw_plot(temp_df, feature, 'figures/' + repo + '_' + feature + '_by_efficiency_box.png', y_label=feature, title=repo,
                      _type='box', use_efficiency=True)

    features = ['sequence_len', 'fix_time']
    for feature in features:
        draw_plot(df, feature, 'figures/' + feature + '_by_efficiency_violin_mixrepo.png', y_label=feature, title=None,
                  _type='violin', use_efficiency=True)
        draw_plot(df, feature, 'figures/' + feature + '_by_efficiency_box_mixrepo.png', y_label=feature, title=None,
                  _type='box', use_efficiency=True)


def set_efficiency_level(read_path):
    # df = pd.read_csv(read_path)



    # for repo in ['tensorflow', 'ansible']:
    #     for cluster in ['0', '1', '2']:
    #         _mean = df.loc[(df['repo_name'].isin([repo])) & (df['cluster'].isin([cluster])), 'efficiency_2'].mean()
    #         df.loc[(df['repo_name'].isin([repo])) & (df['cluster'].isin([cluster])) & (df['efficiency_2'] < _mean), 'efficiency_level'] = 'high'
    #         df.loc[(df['repo_name'].isin([repo])) & (df['cluster'].isin([cluster])) & (
    #                     df['efficiency_2'] >= _mean), 'efficiency_level'] = 'low'
    # df.to_csv(read_path)
    df = pd.read_csv(read_path)
    for cluster in [0, 1, 2]:
        _split = df.loc[(df['cluster'].isin([cluster])), 'fix_efficiency'].mean()
        print(_split)
        df.loc[ (df['cluster'].isin([cluster])) & (df['fix_efficiency'] < _split), 'efficiency_level'] = 'high'
        df.loc[ (df['cluster'].isin([cluster])) & (df['fix_efficiency'] >= _split), 'efficiency_level'] = 'low'
    df.to_csv(read_path, index=0)


if __name__ == '__main__':
    # reset_cluster_name('data/closed_bug_issue_features.json', 'data/closed_bug_issue_features.json')
    # translate_to_csv('data/closed_bug_issue_features.json', 'data/clusters_features.csv')
    # set_efficiency_level('data/clusters_features.csv')
    # generate_all_pics('data/clusters_features.csv')

    # df = pd.read_csv('data/clusters_features.csv')
    # temp_df = df.loc[(df['cluster'] == 2), ['fix_time', 'fix_efficiency', 'sequence_len']]
    # print(temp_df.corr(method='spearman'))

    df = pd.read_csv('data/clusters_features.csv')
    # df = df.loc[:, ['fix_efficiency', 'fix_time', 'sequence_len', 'loc']]
    # print(df.corr(method='spearman'))
    feature = 'sequence_len'

    temp_df = df[df['repo_name'].isin(['ansible'])]
    draw_plot(temp_df, feature, 'figures/ansible_' + feature + '_by_efficiency_box.png', y_label=feature,
              title='ansible',
              _type='box', use_efficiency=True)
    temp_df = df[df['repo_name'].isin(['tensorflow'])]
    draw_plot(temp_df, feature, 'figures/tensorflow_' + feature + '_by_efficiency_box.png', y_label=feature,
              title='tensorflow',
              _type='box', use_efficiency=True)
    draw_plot(df, feature, 'figures/' + feature + '_by_efficiency_box_mixrepo.png', y_label=feature,
              title=None,
              _type='box', use_efficiency=True)

    # f, ax = plt.subplots(figsize=(11, 6))
    # repo = 'tensorflow'
    # temp_df = df[df['repo_name'].isin([repo])]
    # ax.set(ylim=(0, 400000))
    # sns.violinplot(data=temp_df[feature], palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True)
    #
    # # sns.boxplot(x='repo_name', y=feature, data=df, palette="Set3", showmeans=True)
    # plt.savefig('figures/temp.png')

    # draw_plot(df, feature, 'figures/' + feature + '_by_efficiency_box.png', y_label=feature, title=None,
    #           _type='box', use_efficiency=True)

    # _number = 0
    # df1 = pd.read_csv('data/clusters_features_2.csv')
    # df2 = pd.read_csv('data/clusters_features_limited_2.csv')
    # res_list = [[], [], [], [], []]
    #
    # for i in range(len(df1)):
    #     repo_name = df1.loc[i, 'repo_name']
    #     number = df1.loc[i, 'number']
    #     loc1 = df1.loc[i, 'loc']
    #     loc2 = df2.loc[(df2['repo_name'] == repo_name) & (df2['number'] == number), 'loc'].count()
    #     res_list[0].append(repo_name)
    #     res_list[1].append(number)
    #     res_list[2].append(loc1)
    #     if loc2 > 0:
    #         loc2 = df2.loc[(df2['repo_name'] == repo_name) & (df2['number'] == number)]
    #         loc2 = loc2['loc'].tolist()[0]
    #         res_list[3].append(loc2)
    #         d_loc = loc1 - loc2
    #         res_list[4].append(d_loc)
    #     else:
    #         res_list[3].append(0)
    #         res_list[4].append(1)
    #
    #     # print(type(loc1), type(loc2))
    #     # print(repo_name, number, loc1, loc2)
    #     _number = _number + 1
    #
    # df = pd.DataFrame({'repo_name': res_list[0], 'number': res_list[1], 'old_loc': res_list[2], 'new_loc': res_list[3], 'loc_difference_ratio': res_list[4]})
    # df.to_csv('data/clusters_features_comparison_2.csv', index=False)
    #
    # df = pd.read_csv('data/clusters_features_comparison_2.csv')
    # # print(df.loc[:, 'old_loc'])
    # f, ax = plt.subplots(figsize=(11, 6))
    #
    # ax.set(ylim=(0, 700))
    # temp = df[['old_loc', 'new_loc']]
    # temp_df = temp[df['repo_name'].isin(['tensorflow'])]
    # sns.boxplot(data=temp_df, palette="Set3", showmeans=True)
    # # sns.violinplot(data=temp, palette="Set3", bw=.2, cut=1, linewidth=1, showmeans=True)
    # # plt.show()
    # plt.title('tensorflow')
    # plt.savefig('figures/loc_tensorflow.png', dpi=150)
    # sns.despine(left=True, bottom=True)
    # plt.ylabel(y_label)

    # plt.legend(title="efficiency_level", loc=2)
    # plt.savefig(save_path, dpi=150)
