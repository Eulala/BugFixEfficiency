import random
import preprocess
from util import *
from output import *


def calculate_issue_loc():
    abnormal = 0
    res = {}
    bug_fix = load_json_list(preprocess.data_dir + 'c_bug_fix_with_loc.json')
    for d in bug_fix:
        if d['repo_name'] not in res:
            res[d['repo_name']] = {}
        _id = d['target']['number']
        if d['LOC'] > 0:
            if d['LOC'] > 10000:
                abnormal += 1
                continue
            res[d['repo_name']][_id] = d['LOC']
    print("{} issues' LOC > 10000".format(abnormal))
    return res


def KMedians(data, n_clusters, max_iter):
    centers = random.sample([i for i in range(len(data))], n_clusters)
    centers_value = [data[i] for i in centers]
    new_centers = [0 for i in range(n_clusters)]
    new_centers_value = [data[i] for i in new_centers]
    distances = numpy.zeros((len(data), n_clusters))

    Turn = 0
    max_turn = max_iter

    # while Turn < max_turn and (numpy.array(centers_value) != numpy.array(new_centers_value)).any():
    while Turn < max_turn and (numpy.array(centers_value) != numpy.array(new_centers_value)).any():
        if Turn > 0:
            for i in range(len(new_centers)):
                centers_value[i] = new_centers_value[i]
                centers[i] = new_centers[i]

        if Turn % 100 == 0:
            print("-----------------Progress: {} % {}------------------".format(Turn, max_turn))
        clusters_index = []
        clusters_value = []
        for i in range(n_clusters):
            clusters_index.append([])
            clusters_value.append([])

        for i in range(len(data)):
            for c in range(len(centers)):
                distances[i][c] = abs(data[i]-data[centers[c]])

            # print(distances[i])
            min_index = numpy.argsort(numpy.array(distances[i]))[0]
            clusters_index[min_index].append(i)
            clusters_value[min_index].append(data[i])

        # for i in clusters_value:
        #     print(i)
        #     print('--')
        # exit(-1)
        for i in range(len(clusters_value)):
            try:
                _median = numpy.argsort(numpy.array(clusters_value[i]))[int((len(clusters_value[i]))/2)]
                # print(_median)
                new_centers[i] = _median
                new_centers_value[i] = data[_median]
            except Exception:
                new_centers[i] = centers[i]
        Turn = Turn + 1

    print(centers_value, new_centers_value)
    for i in clusters_value:
        print(len(i))

    labels = {}
    for i in range(len(clusters_index)):
        for j in clusters_index[i]:
            labels[j] = i
    return labels


def cluster_by_complexity(issue_data, write_path):
    issue_map = {}
    data = {}
    for repo in issue_data:
        count = 0
        issue_map[repo] = {}
        data[repo] = []
        for issue in issue_data[repo]:
            issue_map[repo][issue] = count
            data[repo].append(issue_data[repo][issue]['all'])
            count = count + 1

    # print(issue_map)
    # print(data)

    # exit(-1)
    for repo in data:
        k_labels = KMedians(data[repo], n_clusters=3, max_iter=1000)
        for i in issue_map[repo]:
            issue_map[repo][i] = str(k_labels[issue_map[repo][i]])

        # # new_data, new_map = delete_outlier(data[repo], issue_map[repo])
        # data_x = numpy.array(data[repo])
        # data_x = data_x.reshape(-1, 1)
        # # # print(data_x.shape)
        # kmeans = KMeans(n_clusters=3)  # n_clusters:number of cluster
        # kmeans.fit(data_x)
        # # print(kmeans.labels_)
        # count = [0, 0, 0, 0, 0, 0, 0, 0]
        # for i in kmeans.labels_:
        #     count[i] = count[i] + 1
        # print(count)
        # for i in issue_map[repo]:
        #     issue_map[repo][i] = str(kmeans.labels_[issue_map[repo][i]])
        # for i in new_map:
        #     new_map[i] = str(kmeans.labels_[new_map[i]])
        # issue_map[repo] = new_map
    # print(issue_map)
    # exit(-1)
    with open(write_path, 'w') as f:
        f.write(json.dumps(issue_map))


def generate_clusters_features(issue_path, cluster_path, write_path):
    loc = calculate_issue_loc(issue_path)
    issues = load_json_list(issue_path)

    issues_f = {}
    for i in issues:
        if i['repo_name'] not in issues_f:
            issues_f[i['repo_name']] = {}
        try:
            issues_f[i['repo_name']][i['number']] = {'fix_efficiency': i['fix_efficiency'], 'fix_time': i['fix_time'], 'sequence_len': i['sequence_len'], 'loc': loc[i['repo_name']][i['number']]['all']}
        except Exception as e:
            # print(e)
            # print(i['repo_name'], i['number'])
            continue

    i_clusters = load_json_list(cluster_path)[0]
    clusters_f = {}
    n_cluster = 3
    for repo in i_clusters:
        clusters_f[repo] = {}
        for i in range(n_cluster):
            clusters_f[repo][str(i)] = {}
        for i in i_clusters[repo]:
            c = i_clusters[repo][i]
            clusters_f[repo][c][int(i)] = issues_f[repo][int(i)]


    write_json_list([clusters_f], write_path)
