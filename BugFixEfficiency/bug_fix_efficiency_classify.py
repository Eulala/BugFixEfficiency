from util import *
from sklearn.cluster import KMeans

bots = { 'tensorflowbutler', 'google-ml-butler', 'tensorflow-bot' }


def load_clusters(path):
    cluster = load_json_data(path)[0]
    return cluster


def generate_bug_fix_features(read_path, write_path):
    bugfix = {}
    with open(read_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['repo_name'] not in bugfix:
                bugfix[dic['repo_name']] = {}
            bugfix[dic['repo_name']][dic['number']] = dic

    res = {}
    for r in bugfix:
        res[r] = {}
        for b in bugfix[r]:
            res[r][b] = {'dev_cont': []}
            # res[b] = {'fix_time': 0, 'time_gap': []}
            sequence = [{"event_type": "RaiseIssueEvent", "created_at": bugfix[r][b]['created_at'], "actor": bugfix[r][b]['author']}] + bugfix[r][b]['events']
            # res[b]['time_gap'] = generate_sequence_time_gap(sequence)
            res[r][b]['fix_time'] = calculate_fix_time_of_issue(sequence)
            # res[b]['n_developer'] = calculate_the_number_of_developer(sequence)
            res[r][b]['dev_cont'] = calculate_contributions_of_developer(sequence)
            effort = []
            for i in res[r][b]['dev_cont']:
                effort.append(i['n_events']/i['n_time'])
            res[r][b]['fix_efficiency'] = numpy.mean(effort)

    with open(write_path, 'w') as f:
        for r in res:
            for i in res[r]:
                f.write(json.dumps({'number': i, 'repo_name': r, 'efficiency': res[r][i]})+'\n')


def generate_sequence_time_gap(events):
    last_human_activity = 0
    for i in range(len(events) - 1, -1, -1):
        if 'actor' in events[i] and events[i]['actor'] in bots:
            continue
        elif 'author' in events[i] and events[i]['author'] in bots:
            continue
        last_human_activity = i
        break
    # print(last_human_activity, len(events))
    res = []
    for i in range(0, last_human_activity):
        res.append(calculate_delta_t(events[i]['created_at'], events[i+1]['created_at']))
    return res


def calculate_the_number_of_developer(events):
    developers = set()
    for i in range(len(events)):
        try:
            developers.add(events[i]['actor'])
        except Exception:
            developers.add(events[i]['author'])
    developers = developers-bots
    return len(developers)


def calculate_contributions_of_developer(events):
    contribution_time = {}
    for i in range(len(events)):
        try:
            d = events[i]['actor']
        except Exception:
            d = events[i]['author']
        if d in bots:
            continue
        if d not in contribution_time:
            contribution_time[d] = []
        contribution_time[d].append(events[i]['created_at'])
    res = []
    for d in contribution_time:
        n_events = len(contribution_time[d])
        n_time = calculate_delta_t(contribution_time[d][0], contribution_time[d][n_events-1])
        if n_time < 1:
            n_time = 1
        res.append({'n_events': n_events, 'n_time': n_time})

    return res


def calculate_fix_time_of_issue(sequence):
    begin_at = sequence[0]['created_at']
    last_human_activity = 0
    for i in range(len(sequence) - 1, -1, -1):
        if 'actor' in sequence[i] and sequence[i]['actor'] in bots:
            continue
        elif 'author' in sequence[i] and sequence[i]['author'] in bots:
            continue
        last_human_activity = i
        break
    end_at = sequence[last_human_activity]['created_at']
    delta_t = calculate_delta_t(begin_at, end_at)
    return delta_t


def calculate_participant_num_of_issue(sequence):
    actor = set()
    for s in sequence:
        actor.add(s['actor'])
    return len(actor)


def calculate_sequence_length(sequence):
    last_human_activity = 0
    for i in range(len(sequence) - 1, -1, -1):
        if 'actor' in sequence[i] and sequence[i]['actor'] in bots:
            continue
        elif 'author' in sequence[i] and sequence[i]['author'] in bots:
            continue
        last_human_activity = i
        break
    return last_human_activity+1


def sequence_clustering(read_path, write_path):
    data_x = {}
    id_map = {}
    with open(read_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            id_map[dic['class']] = []
            d = []
            for e in dic['features']:
                temp = []
                for feature in e:
                    if feature == 'number':
                        continue
                    temp.append(e[feature])
                id_map[dic['class']].append(e['number'])
                d.append(numpy.array(temp))
            data_x[dic['class']] = d

    id_class = {}
    cluster_center = {}
    for x in data_x:
        kmeans = KMeans(n_clusters=2)  # n_clusters:number of cluster
        kmeans.fit(data_x[x])
        id_class[x] = {}
        for i in range(len(id_map[x])):
            # print(i, id_map[x][i], kmeans.labels_[i])
            id_class[x][id_map[x][i]] = kmeans.labels_[i]
        count = [0, 0]
        for i in kmeans.labels_:
            count[i] = count[i]+1
        print(x, count, kmeans.cluster_centers_)
        cluster_center[x] = kmeans.cluster_centers_

    issue_type = {}
    for x in id_class:
        issue_type[x] = {'class': x, 'clusters': {'high': [], 'low': []}}
        high_label = 0
        if cluster_center[x][0][0] > cluster_center[x][1][0]:
            high_label = 1
        for i in id_class[x]:
            if id_class[x][i] == high_label:
                issue_type[x]['clusters']['high'].append(i)
            else:
                issue_type[x]['clusters']['low'].append(i)

    with open(write_path, 'w') as f:
        for x in issue_type:
            f.write(json.dumps(issue_type[x])+'\n')


def generate_sequence_length(issue_path):
    issues = load_json_data(issue_path)
    res = []
    for i in issues:
        length = calculate_sequence_length(i['events'])
        i['sequence_len'] = length
        res.append(i)

    write_json_data(res, issue_path)


def integrate_issue_with_efficiency(issue_path, efficiency):
    issues = load_json_data(issue_path)

    efficiency_dict = {}
    for i in efficiency:
        if i['repo_name'] not in efficiency_dict:
            efficiency_dict[i['repo_name']] = {}
        efficiency_dict[i['repo_name']][i['number']] = {'efficiency': i['efficiency']['fix_efficiency'], 'fix_time': i['efficiency']['fix_time']}
    res = []
    for i in issues:
        repo = i['repo_name']
        number = i['number']
        i['fix_efficiency'] = efficiency_dict[repo][number]['efficiency']
        i['fix_time'] = efficiency_dict[repo][number]['fix_time']
        res.append(i)

    write_json_data(res, issue_path)





