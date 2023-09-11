import copy
import math
import time

import numpy

from util import *


class Sequence:
    def __init__(self, seq, _next: list, cls):
        self.seq = seq
        self.next = _next
        self.cls = cls  # pos and neg


class Pattern:
    def __init__(self, seq, _next: list, sup=None, gr=None, con=float('inf')):
        if sup is None:
            sup = {'pos': 0, 'neg': 0}
        self.seq = seq
        self.next = _next
        self.sup = sup
        self.gr = gr
        self.con = con


class Node:
    def __init__(self, is_tail=False, val=None, depth=0):
        self.val = val
        self.children = {}
        self.is_tail = is_tail
        self.depth = depth
        self.sup = {'pos': 0, 'neg': 0}


class HashTree:
    def __init__(self):
        self.root = Node()

    def print_tree(self):
        node = self.root
        res = [node]
        while len(res) > 0:
            n = res.pop(0)
            children = list(n.children.keys())
            if len(children) > 0:
                print("element: {}, depth: {}, children: {}, is_tail: {}".format(n.val, n.depth, children, n.is_tail))
            else:
                print("element: {}, depth: {}, is_tail: {}, sup: {}".format(n.val, n.depth, n.is_tail, n.sup))
            for c in children:
                res.append(n.children[c])

    def get_sup(self, s):
        node = self.root
        flag = True
        for i in range(len(s)):
            if s[i] in node.children:
                node = node.children[s[i]]
            else:
                flag = False
        if flag:
            # print(node.val, node.is_tail, node.sup)
            return node.sup
        else:
            raise ValueError('No such node')


class GSPTree(HashTree):
    def __init__(self):
        super().__init__()

    def insert(self, s):
        root = self.root
        cur_node = root
        for i in range(len(s)):
            if s[i] not in cur_node.children:
                cur_node.children[s[i]] = Node(val=s[i], depth=cur_node.depth + 1)
            cur_node = cur_node.children[s[i]]
            cur_node.is_tail = False
        cur_node.is_tail = True


class GDSPTree(HashTree):
    def __init__(self):
        super().__init__()

    def insert(self, s, sup):
        root = self.root
        cur_node = root
        for i in range(len(s)):
            if s[i] not in cur_node.children:
                cur_node.children[s[i]] = Node(val=s[i], depth=cur_node.depth + 1)
                cur_node.children[s[i]].sup = sup
            cur_node = cur_node.children[s[i]]
        cur_node.is_tail = True


def generate_event_id(write_path, ignore_events=set()):
    data_dir = get_global_val('data_dir')
    event_set = set()
    for eff in ['high', 'low']:
        data = load_json_list(data_dir + 'bug_fix_sequences_' + eff + '.json')
        for i in data:
            events = i['action_sequence']
            for e in events:
                if e['event_type'] not in ignore_events:
                    event_set.add(e['event_type'])

    event_id = dict(zip(event_set, range(len(event_set))))

    # mapping the alphabet
    for e in event_id:
        if event_id[e] < 26:
            event_id[e] = chr(ord('A') + event_id[e])
        else:
            event_id[e] = chr(ord('a') + event_id[e] - 26)

    write_json_data(event_id, write_path)


def generate_input_sequence(use_entropy=False):
    use_time = False
    data_dir = get_global_val('data_dir')
    ignore_events = {'LockedEvent'}

    if not os.path.exists(data_dir + 'event_id.json'):
        generate_event_id(data_dir + 'event_id.json', ignore_events=ignore_events)

    event_id = load_event_id(data_dir + 'event_id.json')

    input_sequences = {}
    sequences = {}
    if use_entropy:
        interval_split = load_json_dict(data_dir + 'interval_split_entropy.json')
        # interval_split = load_json_dict(data_dir + 'interval_split_4_entropy.json')
    else:
        interval_split = load_json_dict(data_dir + 'interval_split.json')
    for repo in ['ansible', 'tensorflow']:
        for eff in ['low', 'high']:
            input_sequences[eff] = []
            sequences[eff] = {}
            data = load_json_list(data_dir + 'sequences/bug_fix_sequences_' + repo + '_' + eff + '.json')
            for d in data:
                temp = ''
                occur = None
                last_e = None
                for e in d['action_sequence']:
                    # if e['event_type'] in ignore_events:
                    #     continue
                    if e['event_type'] == 'LockedEvent':
                        break  # end
                    if occur is not None:
                        d_t = calculate_delta_t(occur, e['occur_at'], unit='h')
                        e_id = last_e
                        if use_time:
                            split = interval_split[repo][e_id]
                            temp += set_duration_symbol(d_t, split)
                    occur = e['occur_at']
                    temp += event_id[e['event_type']]
                    last_e = e['event_type']
                if use_time:
                    temp += '='
                i = 0
                cur = []
                if use_time:
                    while i < len(temp):
                        cur.append(temp[i] + temp[i + 1])
                        i += 2
                else:
                    while i < len(temp):
                        cur.append(temp[i])
                        i += 1
                input_sequences[eff].append(cur)
                sequences[eff][d['_id']] = temp
        write_json_dict(input_sequences, data_dir + 'sequences/input_sequences_' + repo + '.json')
        write_json_dict(sequences, data_dir + 'sequences/' + repo + '_sequences_symbol_ver.json')


def cut_sequence(length):
    data_dir = get_global_val('data_dir')+'sequences/'
    for repo in ['ansible', 'tensorflow']:
        data = load_json_dict(data_dir+'event_interval/quartile/input_sequences_'+repo+'.json')
        res = {}
        for i in data:
            res[i] = []
            for d in data[i]:
                if len(d) < length:
                    continue
                res[i].append(d[0:length])
                # res[i].append(d)
        write_json_dict(res, data_dir+'input_sequences_'+repo+'.json')


def set_duration_symbol(t, split):
    symbol = ['+', '-', '*', '.']
    for i in range(len(split)):
        if t < split[i]:
            return symbol[i]
    return symbol[len(symbol)-1]


def load_event_id(path):
    with open(path, 'r') as f:
        dic = json.load(f)
        return dic


def temp_test():
    root = GDSPTree()
    root.insert('A', {'pos': 2, 'neg': 1})
    root.insert('B', {'pos': 2, 'neg': 1})
    # root.insert('AB', {'pos': 2, 'neg': 1})
    p_next = build_next('AB')
    p = Pattern('AB', p_next, {'pos': 2, 'neg': 1})
    root.print_tree()
    calculate_ConRe(root.root, p, 0)
    print(p.con)
    # root.print_tree()


def CDSPM():
    # temp_test()
    # exit(-1)
    # Ck = ['123', '125', '153', '234', '253', '345', '534']
    start = time.time()
    data_dir = get_global_val('data_dir') + 'sequences/'
    for repo in ['ansible', 'tensorflow']:
        data = load_json_dict(data_dir + 'input_sequences_' + repo + '.json')
        # data = {'low': [['A+', 'B-', 'E+', 'C='], ['A+', 'B-', 'C*', 'D='], ['A+', 'D+', 'B-', 'E+', 'C=']],
        #         'high': [['A*', 'B-', 'C='], ['A*', 'B='], ['A=']]}

        D = []
        D_size = {'pos': len(data['high']), 'neg': len(data['low'])}
        print("----------------------  generate next pointers ---------------------")
        for s in data['low']:
            _next = build_next(s)
            seq = Sequence(s, _next, 'neg')
            D.append(seq)
        for s in data['high']:
            _next = build_next(s)
            seq = Sequence(s, _next, 'pos')
            D.append(seq)

        for pattern_type in ['neg', 'pos']:
            # if pattern_type == 'neg':
            #     continue
            if pattern_type == 'pos':
                opposite_type = 'neg'
            else:
                opposite_type = 'pos'
            min_gr = 2
            min_con = 1.1
            for min_sup in range(1, 6):
                min_sup = min_sup / 10
                root_c = GSPTree()
                root_g = GDSPTree()

                k = 1
                print("pattern type: {}. positive sequences: {}, negative sequences: {}, min_sup = {}, min_gr = {}, min_con = {}"
                      .format(pattern_type, D_size['pos'], D_size['neg'], min_sup, min_gr, min_con))
                Ck = GSP_ini(D)
                csp = []
                while len(Ck) > 0:
                    print("k={}, generate candidate item sets C_{}: size = {}".format(k, k, len(Ck)))
                    for i in Ck:
                        root_c.insert(i)

                    for s in D:
                        count_candidates(root_c.root, s, idx=0, depth=k)

                    # root_c.print_tree()
                    # exit(-1)

                    Fk, Fp = get_frequent_pattern(root_c, min_sup=min_sup, D_size=D_size[pattern_type], CK=Ck,
                                                  _type=pattern_type)
                    print("number of length = {} fsp: {}".format(k, len(Fk)))
                    # print(Fk)
                    # exit(-1)
                    Gk = get_csp(min_gr=min_gr, D_size=D_size, Fp=Fp, target_type=pattern_type,
                                 opposite_type=opposite_type)
                    print("number of length = {} csp candidates: {}".format(k, len(Gk)))

                    # for conditional discriminative sequential patterns

                    for p in Gk:
                        p.next = build_next(p.seq)
                        calculate_ConRe(root_g.root, p, 0, pattern_type, opposite_type)
                        if p.con > min_con:
                            csp.append({'seq': p.seq, 'sup': p.sup, 'gr': p.gr, 'con': p.con})
                        root_g.insert(p.seq, p.sup)

                    print("number of length <= {} csp: {}".format(k, len(csp)))

                    Ck = GSP(Fk)
                    k += 1

                write_json_data(csp, data_dir + repo + '_' + pattern_type + '_' + str(min_sup) + '_sup_csp.json')

                end = time.time()
                print("min_sup = {}, Runtime: {}".format(min_sup, end - start))


def build_next(s):
    _next = [0] * len(s)
    ptr = {}
    for i in range(len(s) - 1, -1, -1):
        k = s[i]
        ptr[k] = i
        _next[i] = copy.deepcopy(ptr)
    return _next


def count_candidates(node: Node, s: Sequence, idx, depth):
    if idx >= len(s.seq) or not node.children:
        return
    for c in node.children:
        child = node.children[c]
        key = c
        if key in s.next[idx]:
            if child.is_tail and child.depth == depth:
                child.sup[s.cls] += 1
            skip = s.next[idx][key]
            count_candidates(child, s, skip + 1, depth)


def calculate_ConRe(node: Node, p: Pattern, idx, target_type, opposite_type):
    if idx >= len(p.seq) or not node.children:
        return

    for c in node.children:
        child = node.children[c]
        key = child.val
        if key in p.next[idx]:
            if child.is_tail:
                if child.sup[target_type] * p.sup[opposite_type] == 0:
                    p.con = min(p.con, float('inf'))
                else:
                    temp = (p.sup[target_type] * child.sup[opposite_type]) / (
                                child.sup[target_type] * p.sup[opposite_type])
                    p.con = min(p.con, temp)

            skip = p.next[idx][key]
            calculate_ConRe(child, p, skip + 1, target_type, opposite_type)


def GSP_ini(D):
    res = set()
    for i in D:
        seq = i.seq
        for j in seq:
            res.add(j)

    res_list = []
    for r in res:
        res_list.append([r])
    return res_list


def GSP(Ck):
    res = []
    # joint
    for i in range(len(Ck)):
        for j in range(len(Ck)):
            if Ck[i][1: len(Ck[i])] == Ck[j][0: len(Ck[j]) - 1]:
                # print("p1: {}, p2: {}".format(Ck[i], Ck[j]))
                new_item = Ck[i] + [Ck[j][len(Ck[j]) - 1]]
                # prune
                if is_subsequence_in(new_item, Ck):
                    res.append(new_item)
    return res


def is_subsequence_in(s, Ck):
    for i in range(len(s)):
        temp_s = copy.deepcopy(s)
        temp_s.pop(i)
        if temp_s not in Ck:
            return False
    return True


def get_frequent_pattern(root: HashTree, min_sup, D_size, CK, _type):
    res_FK = []
    res_FP = []
    for p in CK:
        sup = root.get_sup(p)
        if (sup[_type] / D_size) >= min_sup:
            res_FK.append(p)
            res_FP.append(Pattern(p, [], sup=sup))
    return res_FK, res_FP


def get_csp(min_gr, D_size, Fp, target_type, opposite_type):
    res = []
    for p in Fp:
        if p.sup[opposite_type] == 0:
            # print(p, sup)
            p.gr = float('inf')
            res.append(p)
        else:
            gr = (p.sup[target_type] / p.sup[opposite_type]) * (D_size[opposite_type] / D_size[target_type])
            if gr >= min_gr:
                # print(p, sup, gr)
                p.gr = gr
                res.append(p)
    return res


def translate_result():
    data_dir = get_global_val('data_dir')
    event_id = load_event_id(data_dir + 'event_id.json')
    data_dir = data_dir + 'sequences/'
    files = os.listdir(data_dir)
    files = list(filter(lambda x: '_sup_csp' in x, files))
    event_map = {event_id[key]: key for key in event_id.keys()}
    # +: < 7d, -: 7~14d, *:14~28d, .:>=28d
    time_map = {'+': 'T1', '-': 'T2', '*': 'T3', '.': 'T4', '=': 'The end'}

    for file in files:
        data = load_json_data(os.path.join(data_dir, file))
        if len(data) == 0:
            continue
        res = []
        for d in data:
            temp = []
            seq = translate_seq(d['seq'], event_map, time_map)
            temp.append(seq)
            temp.append(d['sup']['pos'])
            temp.append(d['sup']['neg'])
            temp.append(d['gr'])
            temp.append(d['con'])
            res.append(temp)

        file = file.replace("json", "csv")
        res = sorted(res, key=lambda x: x[3], reverse=True)
        with open(os.path.join(data_dir, file), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seq', 'sup_pos', 'sup_neg', 'growth_rate', 'condition_redundancy'])
            for d in res:
                writer.writerow(d)


def translate_seq(s, event_map, time_map):
    s = "".join(s)
    res = ' | '
    for i in s:
        if i in event_map:
            res += event_map[i]
        else:
            res += time_map[i]
        res += ' | '
    return res


def time_discretize():
    data_dir = get_global_val('data_dir')
    if not os.path.exists(data_dir + 'event_interval.json'):
        generate_event_interval()

    event_interval = load_json_dict(data_dir + 'event_interval.json')
    interval_split = {}
    # calculate quartile
    for repo in event_interval:
        interval_split[repo] = {}
        interval = event_interval[repo]
        for e in interval:
            data = [d[0] for d in interval[e]]
            qs = numpy.percentile(data, (25, 50, 75, 100), method='midpoint')
            interval_split[repo][e] = list(qs)

    write_json_dict(interval_split, data_dir + 'interval_split.json')


def time_discretize_2_by_entropy():
    data_dir = get_global_val('data_dir')

    if not os.path.exists(data_dir + 'event_interval.json'):
        generate_event_interval()

    event_interval = load_json_dict(data_dir + 'event_interval.json')
    interval_split = {}
    # information gain
    for repo in ['ansible', 'tensorflow']:
        interval_split[repo] = {}
        for e in event_interval[repo]:
            data = sorted(event_interval[repo][e], key=lambda x: x[0])
            best_split = find_split_by_entropy(data)
            if best_split == -1:
                interval_split[repo][e] = []
            else:
                interval_split[repo][e] = [data[best_split][0]]
    write_json_dict(interval_split, data_dir + 'interval_split_entropy.json')


def find_split_by_entropy(data, sum_p=[], sum_n=[]):
    if len(data) == 0:
        return -1

    _min = data[0][0]
    _max = data[len(data) - 1][0]
    if _min == _max:
        return -1
    label_num = [0, 0, 0, 0]

    if len(sum_p) == len(sum_n) == 0:
        sum_p, sum_n = calcu_prefix_sum(data)
    p_num = sum_p[len(sum_p)-1]
    n_num = sum_n[len(sum_n)-1]
    label_num[2] = p_num
    label_num[3] = n_num

    if label_num[2] == 0 or label_num[3] == 0:  # already divided into 2 classes
        return -1

    H_0 = calculate_entropy(p_num, n_num, len(data))
    IG = calculate_information_gain(label_num, len(data), H_0)
    best_split = 0
    max_I = IG
    i = 0
    while len(data)-1 > i > -1:
        split_i = get_split(data[i+1:len(data)], data[i][0])
        if split_i < 0:
            i = len(data)
        else:
            i = split_i + (i + 1)
        label_num = calculate_label_num(sum_p, sum_n, i)
        # print(data[j][0], data[k][0], data[m][0])
        IG = calculate_information_gain(label_num, len(data), H_0)
        # print(max_I)
        if IG > max_I:
            max_I = IG
            best_split = i
        # print(i, len(data), IG)
    # print(max_I)
    return best_split


# def generate_event_interval_with_class():
#     data_dir = get_global_val('data_dir')
#     files = list(filter(lambda x: 'bug_fix' in x, os.listdir(data_dir + 'sequences')))
#     event_interval = {}
#     for file in files:
#         data = load_json_list(data_dir + 'sequences/' + file)
#         if 'high' in file:
#             _type = 'high'
#         else:
#             _type = 'low'
#         for d in data:
#             events = d['action_sequence']
#             for i in range(len(events) - 1):
#                 delta_t = calculate_delta_t(events[i]['occur_at'], events[i + 1]['occur_at'], unit='h')
#                 _id = events[i]['event_type'] + '-' + events[i + 1]['event_type']
#                 if _id not in event_interval:
#                     event_interval[_id] = []
#                 event_interval[_id].append([delta_t, _type])
#     write_json_dict(event_interval, data_dir + 'event_interval_entropy.json')


def generate_event_interval():
    data_dir = get_global_val('data_dir')
    event_interval = {}
    for repo in ['ansible', 'tensorflow']:
        files = list(filter(lambda x: 'bug_fix' in x and repo in x, os.listdir(data_dir + 'sequences')))
        event_interval[repo] = {}
        for file in files:
            data = load_json_list(data_dir + 'sequences/' + file)
            if 'high' in file:
                _type = 'high'
            else:
                _type = 'low'
            for d in data:
                events = d['action_sequence']
                for i in range(len(events) - 1):
                    delta_t = calculate_delta_t(events[i]['occur_at'], events[i + 1]['occur_at'], unit='h')
                    # _id = events[i]['event_type'] + '-' + events[i + 1]['event_type']
                    _id = events[i]['event_type']
                    if _id not in event_interval[repo]:
                        event_interval[repo][_id] = []
                    event_interval[repo][_id].append([delta_t, _type])
    write_json_dict(event_interval, data_dir + 'event_interval.json')


def time_discretize_by_entropy():
    data_dir = get_global_val('data_dir')

    if not os.path.exists(data_dir + 'event_interval.json'):
        generate_event_interval()

    event_interval = load_json_dict(data_dir + 'event_interval.json')
    interval_split = {}
    # information gain
    for repo in event_interval:
        interval_split[repo] = {}
        for e in event_interval[repo]:
            data = sorted(event_interval[repo][e], key=lambda x: x[0])

            split_i = find_split_by_entropy(data)
            if split_i == -1:
                interval_split[repo][e] = []
            else:
                data_l = data[0:split_i]
                data_r = data[split_i:len(data)]
                split_j = find_split_by_entropy(data_l)
                if split_j == -1:
                    split_j = split_i
                split_k = find_split_by_entropy(data_r)
                if split_k == -1:
                    split_k = split_i
                else:
                    split_k = split_k+split_i

                pos = sorted(list({split_j, split_i, split_k}))
                interval_split[repo][e] = []
                for p in pos:
                    interval_split[repo][e].append(data[p][0])
    write_json_dict(interval_split, data_dir + 'interval_split_entropy.json')


def time_discretize_entropy_auto():
    data_dir = get_global_val('data_dir')

    if not os.path.exists(data_dir + 'event_interval.json'):
        generate_event_interval()

    event_interval = load_json_dict(data_dir + 'event_interval.json')
    interval_split = {}
    # information gain
    for repo in event_interval:
        interval_split[repo] = {}
        for e in event_interval[repo]:
            data = sorted(event_interval[repo][e], key=lambda x: x[0])
            # if e == 'UnlockedEvent':
            #     print(data)
            sum_p, sum_n = calcu_prefix_sum(data)
            p_num = sum_p[len(sum_p) - 1]
            n_num = sum_n[len(sum_n) - 1]
            total_num = len(data)
            H_0 = calculate_entropy(p_num, n_num, total_num)
            split_i = find_split_by_entropy(data, sum_p, sum_n)

            if split_i == -1:
                interval_split[repo][e] = []
            else:
                label_num = calculate_label_num(sum_p, sum_n, split_i)
                IG = calculate_information_gain(label_num, total_num, H_0)
                res = [[IG, [split_i]]]

                data_l = data[0:split_i]
                data_r = data[split_i:len(data)]
                sum_p, sum_n = calcu_prefix_sum(data_l)
                split_j = find_split_by_entropy(data_l, sum_p, sum_n)
                if split_j == -1:
                    split_j = split_i
                else:
                    label_l = calculate_label_num(sum_p, sum_n, split_j)+label_num[2:4]
                    IG = calculate_information_gain(label_l, total_num, H_0)
                    res.append([IG, [split_j, split_i]])

                sum_p, sum_n = calcu_prefix_sum(data_r)
                split_k = find_split_by_entropy(data_r, sum_p, sum_n)
                if split_k == -1:
                    split_k = split_i
                else:
                    label_r = label_num[0:2] + calculate_label_num(sum_p, sum_n, split_k)
                    IG = calculate_information_gain(label_r, total_num, H_0)
                    split_k = split_k + split_i
                    res.append([IG, [split_i, split_k]])

                if split_j != split_i and split_i != split_k:
                    labels = label_l[0:4] + label_r[2:6]
                    IG = calculate_information_gain(labels, total_num, H_0)
                    res.append([IG, [split_j, split_i, split_k]])

                # interval_split[repo][e] = [data[split_j][0], data[split_i][0], data[split_k][0]]
                res = sorted(res, key=lambda x: x[0], reverse=True)
                interval_split[repo][e] = []
                for pos in res[0][1]:
                    interval_split[repo][e].append(data[pos][0])
    write_json_dict(interval_split, data_dir + 'interval_split_auto_entropy.json')


def calcu_prefix_sum(data):
    if len(data) == 0:
        return [], []
    pos = [0] * len(data)
    neg = [0] * len(data)
    for i in range(len(data)):
        if data[i][1] == 'high':
            pos[i] = 1
        else:
            neg[i] = 1
    sum_p = [0] * len(data)
    sum_n = [0] * len(data)
    sum_p[0] = pos[0]
    sum_n[0] = neg[0]
    for i in range(1, len(pos)):
        sum_p[i] = sum_p[i - 1] + pos[i]
        sum_n[i] = sum_n[i - 1] + neg[i]
    return sum_p, sum_n


def get_split(data, ini_val):
    for i in range(len(data)):
        if data[i][0] > ini_val:
            return i
    return -1


def calculate_label_num(sum_p, sum_n, split_pos):
    # start = time.time()
    label_num = [0]*4

    last_p = 0
    last_n = 0
    if split_pos == 0:
        label_num[0] = 0
        label_num[1] = 0
    else:
        label_num[0] = sum_p[split_pos-1]
        label_num[1] = sum_n[split_pos-1]

    label_num[2] = sum_p[len(sum_p)-1] - label_num[0]
    label_num[3] = sum_n[len(sum_n)-1] - label_num[1]
    # end = time.time()
    # print(end-start)
    return label_num


def calculate_entropy(p_num, n_num, total):
    H_L = 0
    H_R = 0
    if p_num != 0:
        H_L = - (p_num / total) * numpy.log(p_num / total)
    if n_num != 0:
        H_R = - (n_num / total) * numpy.log(n_num / total)
    H_0 = H_L + H_R
    return H_0


def calculate_information_gain(label_num, data_num, H):
    # print(label_num)
    _bin = int(len(label_num)/2)
    calcu_part = [0]*len(label_num)
    _num = [0]*_bin
    for i in range(len(label_num)):
        k = math.floor(i / 2)
        _num[k] += label_num[i]

    for i in range(len(label_num)):
        if label_num[i] != 0:
            k = math.floor(i / 2)
            calcu_part[i] = (label_num[i] / _num[k]) * (numpy.log(label_num[i] / _num[k]))

    LH = [0]*_bin
    for i in range(len(label_num)):
        k = math.floor(i / 2)
        LH[k] -= calcu_part[i]

    IG = H
    for i in range(_bin):
        IG -= LH[i] * (_num[i]/data_num)
    return IG

