from util import *
from preprocess import *


def model_state_translation(args):
    # args: issue_with_event, pr_with_event
    issues = set()
    prs = set()
    closed_issue = 0
    with open(args[0], 'r') as f:
        for i in f:
            dic = json.loads(i)
            issues.add(dic['number'])
            for e in dic['events']:
                if e['event_type'] == 'ClosedEvent':
                    closed_issue = closed_issue+1
                    break
    with open(args[1], 'r') as f:
        for i in f:
            dic = json.loads(i)
            prs.add(dic['number'])
    print(len(issues), closed_issue)
    issues_link = get_cross_reference(args[0], prs)
    prs_link = get_cross_reference(args[1], issues)
    issue_pr_map = map_cross_reference(issues_link, prs_link)
    select_pr = set()
    for i in issue_pr_map:
        for j in issue_pr_map[i]:
            # if j in select_pr:
            #     print(i, issue_pr_map[i])
            select_pr.add(j)
    remain_pr = { }
    with open(args[1], 'r') as f:
        for i in f:
            dic = json.loads(i)
            if dic['number'] in select_pr:
                remain_pr[dic['number']] = dic
    # print(len(remain_pr))

    states = generate_state_translation(args[0], remain_pr, issue_pr_map)
    record_state(states)


def generate_state_translation(issue_path, linked_prs, issue_pr_map):
    states = {}
    with open(issue_path,  'r') as f:
        for i in f:
            dic = json.loads(i)
            temp = {'state': [1, 2], 'occur_at': [-1, dic['created_at']], 'action': ['RaiseIssueEvent']}
            current_state = 2
            for e in dic['events']:
                if e['event_type'] == "AssignedEvent" and current_state != 3:
                    current_state = 3
                    temp['state'].append(current_state)
                    temp['occur_at'].append(e['created_at'])
                    temp['action'].append(e['event_type'])
                elif e['event_type'] == 'CrossReferencedEvent':
                    if dic['number'] in issue_pr_map and e['linked_pr'] in issue_pr_map[dic['number']]:
                        current_state = 4
                        temp['state'].append(current_state)
                        temp['occur_at'].append(e['created_at'])
                        temp['action'].append(e['event_type'])
                        res_states = generate_state_of_pr(linked_prs[e['linked_pr']]['events'])
                        temp['state'] = temp['state'] + res_states['state']
                        temp['occur_at'] = temp['occur_at'] + res_states['occur_at']
                        temp['action'] = temp['action'] + res_states['action']
                        current_state = temp['state'][len(temp['state'])-1]
                        if current_state == 8:
                            break
                elif e['event_type'] == 'LabeledEvent' and e['label'] == 'invalid' or e['event_type'] == 'MarkedAsDuplicateEvent':
                    current_state = 8
                    temp['state'].append(current_state)
                    temp['occur_at'].append(e['created_at'])
                    temp['action'].append(e['event_type'])
                    break
                elif e['event_type'] == 'CommentEvent':
                    # TODO
                    continue
                elif e['event_type'] in ['UnassignedEvent', 'ReopenedEvent'] and current_state != 2:
                    current_state = 2
                    temp['state'].append(current_state)
                    temp['occur_at'].append(e['created_at'])
                    temp['action'].append(e['event_type'])
                else:
                    continue
            states[dic['number']] = temp
    return states


def generate_state_of_pr(events):
    temp = {'state': [], 'occur_at': [], 'action': []}
    current_state = 4
    for e in events:
        try:
            res_state, res_occur = state_func_dict.get(current_state, func_none)(e)
            if res_state == -1:
                continue
            else:
                current_state = res_state
                temp['state'].append(res_state)
                temp['occur_at'].append(res_occur)
                temp['action'].append(e['event_type'])
        except Exception:
            break
    return temp


def generate_next_state_from_4(event):
    if event['event_type'] == 'AssignedEvent':
        next_state = 5
    else:
        next_state = -1
    # else:
    #     print("4, {}".format(event))
    #     exit(-1)
    #     next_state = -1

    next_occur = event['created_at']
    return next_state, next_occur


def generate_next_state_from_5(event):
    if event['event_type'] in ['MergedEvent', 'BaseRefForcePushedEvent', 'HeadRefForcePushedEvent']:
        next_state = 6
    elif event['event_type'] == 'ClosedEvent':
        next_state = 7
    else:
        next_state = -1
    next_occur = event['created_at']
    return next_state, next_occur


def generate_next_state_from_6(event):
    if event['event_type'] == 'ClosedEvent':
        next_state = 8
    else:
        next_state = -1
    # else:
    #     print("5, {}".format(event))
    #     exit(-1)
    #     next_state = -1

    next_occur = event['created_at']
    return next_state, next_occur


def generate_next_state_from_7(event):
    # TODO: raise an issue 7->2
    if event['event_type'] == 'ReopenedEvent':
        next_state = 5
    else:
        next_state = -1
    next_occur = event['created_at']
    return next_state, next_occur


def record_state(states):
    res = []
    for s in states:
        dic = states[s]
        dic['number'] = s
        res.append(dic)
    write_json_list(res, 'data/bug_state.json')


state_func_dict = { 4: generate_next_state_from_4, 5: generate_next_state_from_5, 6: generate_next_state_from_6, 7: generate_next_state_from_7}


def calculate_translate_probability(bug_state_path):
    state_translation_frequency = {}
    for i in range(1, 8):
        state_translation_frequency[i] = {}
    with open(bug_state_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            n_state = len(dic['state'])
            if n_state < 2:
                continue
            for j in range(0, n_state-1):
                if dic['state'][j+1] not in state_translation_frequency[dic['state'][j]]:
                    state_translation_frequency[dic['state'][j]][dic['state'][j+1]] = 1
                else:
                    state_translation_frequency[dic['state'][j]][dic['state'][j + 1]] = state_translation_frequency[dic['state'][j]][dic['state'][j + 1]] + 1
    print(state_translation_frequency)


def calculate_translate_cost(bug_state_path):
    state_translation_cost = { }
    for i in range(1, 8):
        state_translation_cost[i] = {}
    with open(bug_state_path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            n_state = len(dic['state'])
            if n_state < 3:
                continue
            for j in range(1, n_state-1):
                delta_t = calculate_delta_t(dic['occur_at'][j], dic['occur_at'][j+1])
                if delta_t < 0:
                    continue
                if dic['state'][j+1] not in state_translation_cost[dic['state'][j]]:
                    state_translation_cost[dic['state'][j]][dic['state'][j+1]] = []
                state_translation_cost[dic['state'][j]][dic['state'][j + 1]].append(delta_t)
    for s1 in state_translation_cost:
        for s2 in state_translation_cost[s1]:
            state_translation_cost[s1][s2] = numpy.mean(state_translation_cost[s1][s2])
    print(state_translation_cost)


def generate_state_action_sequence(path):
    res = []
    with open(path, 'r') as f:
        for i in f:
            dic = json.loads(i)
            temp_list = [dic['state'][0]]
            for j in range(0, len(dic['action'])):
                temp_list.append(dic['action'][j])
                temp_list.append(dic['state'][j+1])
            res.append(temp_list)
    write_json_list(res, 'data/bug_life_sequences.json')


def generate_event_transition_matrix(path, write_path):
    event_sequences = load_json_list(path)
    event_id = load_json_list('data/event_id.json')[0]
    # if is_efficient:
    #     event_id = load_json_data('data/efficient_event_id.json')[0]
    # else:
    #     event_id = load_json_data('data/inefficient_event_id.json')[0]

    transition_matrix_set = {}
    for i in event_sequences:
        transition_matrix = [[0] * len(event_id) for j in range(len(event_id))]
        row_id = event_id['IssueReportEvent']
        for e in range(len(i['events'])):
            col_id = event_id[i['events'][e]['event_type']]
            transition_matrix[row_id][col_id] = transition_matrix[row_id][col_id] + 1
            row_id = event_id[i['events'][e]['event_type']]
        transition_matrix_set[i['number']] = transition_matrix

    with open(write_path, 'w') as f:
        for i in transition_matrix_set:
            f.write(json.dumps({'number': i, 'matrix': transition_matrix_set[i]})+'\n')
    # write_json_data(transition_matrix_set, write_path)
    return transition_matrix_set


def matrix_clustering(matrix, write_path):
    data_x = []
    num_map = {}
    num_count = 0
    for i in matrix:
        num_map[i] = num_count
        A = numpy.array(matrix[i])
        A = A.reshape(1, len(A[0]) * len(A[0]))
        # print(A.shape)
        A = A.tolist()
        # print(A)
        data_x.append(A[0])
        num_count = num_count+1
    # print(data_x)
    data_x = numpy.array(data_x)
    print(data_x.shape)

    kmeans = KMeans(n_clusters=3)  # n_clusters:number of cluster
    kmeans.fit(data_x)
    # print(kmeans.labels_)
    count = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in kmeans.labels_:
        count[i] = count[i]+1
    print(count)
    for i in num_map:
        num_map[i] = str(kmeans.labels_[num_map[i]])

    with open(write_path, 'w') as f:
        f.write(json.dumps(num_map))
