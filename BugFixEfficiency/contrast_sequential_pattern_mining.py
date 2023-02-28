from util import *


def generate_event_id(read_path, write_path):
    event_sequences_set = []
    for r in read_path:
        for d in load_json_list(r):
            event_sequences_set = event_sequences_set + d['events']

    event_types = set()
    for event in event_sequences_set:
        event_types.add(event['event_type'])

    event_id = dict(zip(event_types, range(len(event_types))))

    # mapping the alphabet
    for e in event_id:
        if event_id[e] < 26:
            event_id[e] = chr(ord('A')+event_id[e])
        else:
            event_id[e] = chr(ord('a')+event_id[e]-26)
    with open(write_path, 'w') as f:
        f.write(json.dumps(event_id))


def generate_input_sequence(read_path: list, write_path):
    # read_path: arg1 = cluster_features_path, arg2 = issue_path (with events), arg3 = event_id_path
    clusters = pd.read_csv(read_path[0])
    bug_issues = load_json_list(read_path[1])
    event_id = {}
    with open(read_path[2], 'r') as f:
        for i in f:
            event_id = json.loads(i)

    bug_fix = {}
    for i in bug_issues:
        number = i['number']
        repo = i['repo_name']
        try:
            cluster = clusters.loc[(clusters['repo_name'] == repo) & (clusters['number'] == number)]
            cluster = cluster['cluster'].tolist()[0]
            e_level = clusters.loc[(clusters['repo_name'] == repo) & (clusters['number'] == number)]
            e_level = e_level['efficiency_level'].tolist()[0]
            if cluster not in bug_fix:
                bug_fix[cluster] = {}
            if e_level not in bug_fix[cluster]:
                bug_fix[cluster][e_level] = []
            temp = {'number': number, 'repo_name': repo, 'sequence': generate_event_sequence(i['events'], event_id)}
            bug_fix[cluster][e_level].append(temp)
        except Exception:
            continue

    write_json_list([bug_fix], write_path)


def generate_event_sequence(events, event_id):
    res = ''
    for e in events:
        if e['event_type'] == 'I_ReportedEvent':
            continue
        _id = event_id[e['event_type']]
        res = res + _id
    return res


def mining_CSP(read_path, min_cr=1):
    read_data = load_json_list(read_path)[0]
    for i in read_data:
        print("cluster: {}, sequence_num: high={}, low={}".format(i, len(read_data[i]['high']), len(read_data[i]['low'])))
        data_1 = []
        data_2 = []
        for s in read_data[i]['high']:
            data_1.append(s['sequence'])
        for s in read_data[i]['low']:
            data_2.append(s['sequence'])
        CSPs = generate_CSP(data_1, data_2, min_cr)

        # CSP_str = []
        # CR = {}
        # for j in CSPs:
        #     CSP_str.append(j['CSP'])
        #     CR[j['CSP']] = j['CR']
        # if_contain_subsequence(CSP_str, CR)

        with open('data/' + str(i) + 'CSP_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['CSP', 'CR', 'Sup_1', 'Sup_2', 'C1_num', 'C2_num', 'Class'])
            for j in CSPs:
                writer.writerow([j['CSP'], j['CR'], j['Sup_1'], j['Sup_2'], j['C1_num'], j['C2_num'], j['class']])
                # f.write(json.dumps(j) + '\n')


def remove_subsequence_csp(data_path, write_path):
    sequences = set()
    data = []
    head = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'CSP':
                head = row
                continue
            sequences.add(row[0])
            data.append(row)
    remained = delete_subsequence(sequences)

    with open(write_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for d in data:
            if d[0] in remained:
                writer.writerow(d)


def delete_subsequence(sequences):
    delete_set = set()
    for target in sequences:
        flag = False
        for source in sequences:
            if target == source:
                continue
            cur_pos_t = 0
            cur_pos_s = 0
            while cur_pos_s < len(source):
                if target[cur_pos_t] == source[cur_pos_s]:
                    cur_pos_t = cur_pos_t + 1
                cur_pos_s = cur_pos_s + 1
                if cur_pos_t == len(target):
                    flag = True
                    break

            if flag is True:
                # target is a subsequence of source
                delete_set.add(target)
                # if CR[target] == -1 and CR[source] > 0:
                #     print(target, source)
    print(len(delete_set), len(sequences))
    return sequences-delete_set


def load_event_id(path='data/event_id.json'):
    res = set()
    with open(path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            for e in dic:
                if dic[e] == 'j':
                    continue
                res.add(dic[e])
    return res


def generate_CSP(data_1, data_2, min_cr):
    # Initialize
    events = load_event_id()
    min_sup = int(0.05 * (len(data_1) + len(data_2)))
    # min_sup_1 = int(0.05 * len(data_1))
    # min_sup_2 = int(0.05 * len(data_2))
    frequent_one_item_set = generate_frequent_one_item(data_1+data_2, min_sup, events)    # Find all frequent 1-item patterns

    D_num_1 = len(data_1)
    D_num_2 = len(data_2)


    CSPs = []
    # Recursion
    for item in frequent_one_item_set:
        DFS_CSP(item, item, min_sup, data_1, data_2, D_num_1, D_num_2, frequent_one_item_set, CSPs, min_cr)
    return CSPs


def generate_frequent_one_item(data, min_sup, events):
    res = set()
    for e in events:
        sup = calculate_sup(e, data)
        if sup >= min_sup:
            res.add(e)
    return res


def calculate_sup(event, sequences):
    count = 0
    for s in sequences:
        if event in s:
            count = count + 1
    return count


def generate_project_database(item, sequences):
    res = []
    for s in sequences:
        if len(s) > 0:
            proj = generate_project_sequence(item, s)
            res.append(proj)
    return res


def generate_project_sequence(item, sequence: str):
    cur_pos_s = sequence.find(item)
    if cur_pos_s >= 0:
        return sequence[cur_pos_s: len(sequence)]
    else:
        return []


def generate_new_prefix(prev, addition):
    return prev+addition


def calculate_notnull_count(data: list):
    count = 0
    for s in data:
        if len(s) > 0:
            count = count + 1
    return count


def calculate_CR(sup_1, sup_2, D_1, D_2):
    if sup_1 == sup_2 == 0:
        return 0, 0
    if sup_1 == 0:
        CR = -1
        _class = 2
    elif sup_2 == 0:
        CR = -1
        _class = 1
    else:
        GR_1 = (sup_1 / D_1) / (sup_2 / D_2)
        if GR_1 >= 1:
            CR = GR_1
            _class = 1
        else:
            CR = 1 / GR_1
            _class = 2
    return CR, _class


def calculate_chi_square(sup_x_1, sup_x_2, sup_y_1, sup_y_2):
    chi = ((sup_x_1+sup_x_2+sup_y_1+sup_y_2)*math.pow((sup_x_1*sup_y_2 - sup_x_2*sup_y_1), 2))/((sup_x_1+sup_x_2)*(sup_y_1+sup_y_2)*(sup_x_1+sup_y_1)*(sup_x_2+sup_y_2))
    return chi


def DFS_CSP(current_e, item, min_sup, proj_1, proj_2, D_1, D_2, one_item_set, CSPs, min_cr):
    # calculate support
    sup_1 = calculate_sup(current_e, proj_1)
    sup_2 = calculate_sup(current_e, proj_2)
    # print(current_e, sup_1, sup_2)

    # generate corresponding projected database
    d_proj_1 = generate_project_database(current_e, proj_1)
    d_proj_2 = generate_project_database(current_e, proj_2)

    # calculate the size of corresponding projected database
    len_c1 = calculate_notnull_count(d_proj_1)
    len_c2 = calculate_notnull_count(d_proj_2)

    # if len_c1 > 0.5*min_sup and len_c1 > len_c2:
    #     print(len_c1, len_c2)

    # calculate CR
    if sup_1 + sup_2 >= 0.01*(D_1+D_2):
        CR, _class = calculate_CR(sup_1, sup_2, D_1, D_2)
        if CR == -1 or CR >= min_cr:
            CSPs.append({ 'CSP': item, 'CR': CR, 'Sup_1': sup_1, 'Sup_2': sup_2, 'C1_num': D_1, 'C2_num': D_2,
                          'class': _class })

    if len_c1 == 0 or len_c2 == 0 or sup_1 == 0 or sup_2 == 0:  # no child in one class or any support of the two classes equals 0
        return

    if sup_1+sup_2 <= min_sup:  # sup(prefix) < min_sup
        return

    for i in one_item_set:
        new_item = generate_new_prefix(item, i)
        # pruning
        sup_y_1 = calculate_sup(i, d_proj_1)
        sup_y_2 = calculate_sup(i, d_proj_2)
        if sup_y_1 + sup_y_2 > 0:
            chi_square = calculate_chi_square(sup_1, sup_2, sup_y_1, sup_y_2)
            if chi_square < 1:
                continue
        # print(current_e, i, sup_y_1, sup_y_2)
        # exit(-1)
        DFS_CSP(i, new_item, min_sup, d_proj_1, d_proj_2, D_1, D_2, one_item_set, CSPs, min_cr)




    





























