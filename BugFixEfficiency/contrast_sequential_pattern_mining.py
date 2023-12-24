import copy
import math
import os
import pickle
import random
import time

import numpy
from termcolor import colored

from util import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


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


def generate_event_id(repo_name, write_path, ignore_events=set()):
    data_dir = get_global_val('data_dir')
    if repo_name == 'total':
        files = list(filter(lambda x: '_closed_issues' + '.json' in x, os.listdir(data_dir)))
        data = []
        for f in files:
            data += load_json_list(os.path.join(data_dir, f))
    else:
        data = load_json_list(os.path.join(data_dir, repo_name + '_closed_issues.json'))
    event_set = set()

    for i in data:
        events = i['action_sequence']
        # events = i
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


def model_sequence(repo_name):
    data_dir = os.path.join(get_global_val('data_dir'), repo_name)
    event_id = load_event_id(os.path.join(data_dir, 'event_id.json'))

    # files = os.listdir(data_dir)
    files = ['issue_sequences_neg.json', 'issue_sequences_pos.json', 'issue_sequences_med.json']
    for f in files:
        res = []
        data = load_json_list(os.path.join(data_dir, f))
        for d in data:
            temp = {'_id': d['_id'], 'action_sequence': []}
            last_t = None
            for a in d['action_sequence']:
                # if a['event_type'] in ignore_events:
                #     break
                e_id = event_id[a['event_type']]
                cur_t = a['occur_at']
                if last_t is not None:
                    delta_t = calculate_delta_t(last_t, cur_t, unit='m')
                    temp['action_sequence'].append(delta_t)
                last_t = cur_t
                temp['action_sequence'].append(e_id)
            res.append(temp)
        file = f.split('.')[0]
        write_json_list(res, os.path.join(data_dir, file+'_model.json'))


def generate_input_sequence(write_dir, use_entropy=False, file_suffix=None):
    data_dir = get_global_val('data_dir') + 'sequences/'
    input_sequences = {}
    sequences = {}
    interval_dir = os.path.abspath(os.path.join(write_dir, ".."))
    if use_entropy:
        interval_split = load_json_dict(os.path.join(interval_dir, 'interval_split_auto_entropy.json'))
        # interval_split = load_json_dict(data_dir + 'interval_split_4_entropy.json')
    else:
        interval_split = load_json_dict(os.path.join(interval_dir,  'interval_split.json'))
    for repo in ['ansible', 'tensorflow']:
        for eff in ['pos', 'neg']:
            input_sequences[eff] = []
            sequences[eff] = {}
            data = load_json_list(data_dir + 'issue_sequences_' + repo + '_' + eff + file_suffix+'.json')
            for d in data:
                i = 0
                temp = ''
                while i < len(d['action_sequence']):
                    e = d['action_sequence'][i]
                    t = d['action_sequence'][i+1]
                    split = interval_split[repo][e]
                    temp += e
                    temp += set_duration_symbol(t, split)

                    i += 2
                cur = []
                i = 0
                while i < len(temp):
                    cur.append(temp[i] + temp[i + 1])
                    i += 2
                input_sequences[eff].append(cur)
                sequences[eff][d['_id']] = temp

        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        write_json_dict(input_sequences, write_dir + 'input_sequences_' + repo + '.json')
        write_json_dict(sequences, write_dir + repo + '_sequences_symbol_ver.json')


def generate_input_sequence_ete(x_train, y_train, write_dir, filename, use_entropy=False):
    # interval_dir = os.path.abspath(os.path.join(write_dir, ".."))
    input_sequences = {'pos': [], 'neg': []}
    # input_sequences = {'pos': [], 'neg': [], 'med': []}
    sequences = {}
    if use_entropy:
        interval_split = load_json_dict(os.path.join(write_dir, 'interval_split_auto_entropy.json'))
    else:
        interval_split = load_json_dict(os.path.join(write_dir,  'interval_split.json'))

    for d in range(len(x_train)):
        i = 0
        temp = ''
        seq = x_train[d]
        eff = y_train[d]
        while i < len(seq)-1:
            e1 = seq[i]
            t = seq[i + 1]
            e2 = seq[i+2]
            try:
                split = interval_split[e1+'_'+e2]
            except Exception:
                split = []
            temp += e1
            temp += set_duration_symbol(t, split)
            # temp += '+'
            # if e1 == 'E' and e2 == 'H':
            #     print(t, split, set_duration_symbol(t, split))
            i += 2
        temp += seq[len(seq)-1]
        i = 0
        cur = []
        while i < len(temp)-1:
            try:
                cur.append(temp[i] + temp[i + 1] + temp[i+2])
                i += 2
            except Exception:
                print(temp, i)
                exit(-1)
        # if cur[len(cur)-1][2] == 'B':
        #     # delete the last closed event
        #     cur = cur[0:len(cur)-1]
        input_sequences[eff].append(cur)
    write_json_dict(input_sequences, os.path.join(write_dir, filename))


def generate_med_sequence_ete(read_file, write_dir, filename, use_entropy=False):
    # interval_dir = os.path.abspath(os.path.join(write_dir, ".."))
    input_sequences = []
    if use_entropy:
        interval_split = load_json_dict(os.path.join(write_dir, 'interval_split_auto_entropy.json'))
    else:
        interval_split = load_json_dict(os.path.join(write_dir,  'interval_split.json'))

    data = load_json_list(read_file)
    for d in data:
        i = 0
        temp = ''
        seq = d['action_sequence']
        while i < len(seq)-1:
            e1 = seq[i]
            t = seq[i + 1]
            e2 = seq[i+2]
            try:
                split = interval_split[e1+'_'+e2]
            except Exception:
                split = []
            temp += e1
            temp += set_duration_symbol(t, split)
            # temp += '+'
            # if e1 == 'E' and e2 == 'H':
            #     print(t, split, set_duration_symbol(t, split))
            i += 2
        temp += seq[len(seq)-1]
        i = 0
        cur = []
        while i < len(temp)-1:
            try:
                cur.append(temp[i] + temp[i + 1] + temp[i+2])
                i += 2
            except Exception:
                print(temp, i)
                exit(-1)
        # if cur[len(cur)-1][2] == 'B':
        #     # delete the last closed event
        #     cur = cur[0:len(cur)-1]
        input_sequences.append(cur)
    write_json_dict(input_sequences, os.path.join(write_dir, filename))


def generate_input_sequence_e(x_train, y_train, write_dir, filename):
    input_sequences = {'pos': [], 'neg': []}
    sequences = {}

    for d in range(len(x_train)):
        i = 0
        seq = x_train[d]
        eff = y_train[d]
        cur = []
        while i < len(seq):
            cur.append(seq[i])
            i += 2
        input_sequences[eff].append(cur)
    write_json_dict(input_sequences, os.path.join(write_dir, filename))

def generate_input_sequence_ee(x_train, y_train, write_dir, filename):
    input_sequences = {'pos': [], 'neg': []}

    for d in range(len(x_train)):
        i = 0
        seq = x_train[d]
        eff = y_train[d]
        cur = []
        while i < len(seq)-1:
            cur.append(seq[i]+seq[i+2])
            i += 2
        input_sequences[eff].append(cur)
    write_json_dict(input_sequences, os.path.join(write_dir, filename))


def generate_all_sequence_ete(data, write_dir, filename, use_entropy=False):
    # interval_dir = os.path.abspath(os.path.join(write_dir, ".."))
    if use_entropy:
        interval_split = load_json_dict(os.path.join(write_dir, 'interval_split_auto_entropy.json'))
    else:
        interval_split = load_json_dict(os.path.join(write_dir,  'interval_split.json'))

    res = []
    for d in data:
        i = 0
        seq = d['seq']
        temp = ''
        while i < len(seq)-1:
            e1 = seq[i]
            t = seq[i + 1]
            e2 = seq[i+2]
            try:
                split = interval_split[e1+'_'+e2]
            except Exception:
                split = []
            temp += e1
            temp += set_duration_symbol(t, split)
            # temp += '+'
            # if e1 == 'E' and e2 == 'H':
            #     print(t, split, set_duration_symbol(t, split))
            i += 2
        temp += seq[len(seq)-1]
        i = 0
        cur = []
        while i < len(temp)-1:
            try:
                cur.append(temp[i] + temp[i + 1] + temp[i+2])
                i += 2
            except Exception:
                print(temp, i)
                exit(-1)
        d['seq'] = cur
        res.append(d)
    write_json_list(res, os.path.join(write_dir, filename))


def cut_sequence(min_len, max_len, max_t, repo_name, interval=2):
    data_dir = get_global_val('data_dir')+repo_name

    max_time = max_t
    cut_len = 0
    if interval == 2:
        cut_len = max_len*2
    elif interval == 3:
        cut_len = max_len*2+1
    files = os.listdir(data_dir)
    files = list(filter(lambda x: '_model' in x, files))
    for file in files:
        res = []
        data = load_json_list(os.path.join(data_dir, file))
        for d in data:
            if len(d['action_sequence']) < min_len*2+1:
                # print(d)
                continue
            seq = d['action_sequence'][0:cut_len]

            k = 1
            sum_t = 0
            idx = k
            while k < len(seq) - 1:
                if sum_t >= max_time:
                    idx = k
                    break
                sum_t += d['action_sequence'][k]
                k += 2
            if idx == 0:  # less than max days
                temp = {'_id': d['_id'], 'action_sequence': seq}
            else:
                temp = {'_id': d['_id'], 'action_sequence': seq[0:k]}
            if len(temp['action_sequence']) < min_len*2+1:
                continue
            # temp = {'_id': d['_id'], 'action_sequence': seq}
            res.append(temp)

        file = file.replace("model", 'cut_origin')
        write_json_list(res, os.path.join(data_dir, file))


def cut_sequence_by_time(day, repo_name, interval=2):
    data_dir = get_global_val('data_dir') + repo_name
    files = os.listdir(data_dir)
    files = list(filter(lambda x: '_model' in x, files))
    for file in files:
        res = []
        data = load_json_list(os.path.join(data_dir, file))
        for d in data:
            k = 1
            sum_t = 0
            idx = k
            while k < len(d['action_sequence'])-1:
                if sum_t >= day*60*24:
                    idx = k
                    break
                sum_t += d['action_sequence'][k]
                k += 2
            if idx == 0:  # not long enough
                continue

            temp = {'_id': d['_id'], 'action_sequence': d['action_sequence'][0:k]}
            if len(temp['action_sequence']) < 10:
                continue
            res.append(temp)

        file = file.replace("model", 'cut_origin')
        write_json_list(res, os.path.join(data_dir, file))

def set_duration_symbol(t, split):
    symbol = ['+', '-', '*', '.']

    for i in range(len(split)):
        if int(t) < int(split[i]):
            return symbol[i]

    return symbol[len(split)]


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


def CDSPM(data_dir, count, min_gr=1.5):
    # temp_test()
    # exit(-1)
    # Ck = ['123', '125', '153', '234', '253', '345', '534']
    start = time.time()

    data = load_json_dict(os.path.join(data_dir, 'input_sequences_'+str(count)+'.json'))
    # data = {'low': [['A+', 'B-', 'E+', 'C='], ['A+', 'B-', 'C*', 'D='], ['A+', 'D+', 'B-', 'E+', 'C=']],
    #         'high': [['A*', 'B-', 'C='], ['A*', 'B='], ['A=']]}

    D = []
    D_size = {'pos': len(data['pos']), 'neg': len(data['neg'])}
    print("----------------------  generate next pointers ---------------------")
    for s in data['neg']:
        _next = build_next(s)
        seq = Sequence(s, _next, 'neg')
        D.append(seq)
    for s in data['pos']:
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
        min_con = 1.0
        for min_sup in range(1, 2):
            min_sup = min_sup / 10
            # min_sup = 0.05
            root_c = GSPTree()
            root_g = GDSPTree()

            k = 1
            print("pattern type: {}. positive sequences: {}, negative sequences: {}, min_sup = {}, min_gr = {}, min_con = {}"
                  .format(pattern_type, D_size['pos'], D_size['neg'], min_sup, min_gr, min_con))
            Ck = GSP_ini(D)
            csp = []
            fsp = []
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
                for p in Fp:
                    fsp.append({'seq': p.seq, 'sup': {'pos': p.sup['pos']/D_size['pos'], 'neg': p.sup['neg']/D_size['neg']}})

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

            write_json_data(csp, os.path.join(data_dir, pattern_type + '_' + str(min_sup) + '_sup_csp_'+str(count)+'.json'))
            write_json_data(fsp, os.path.join(data_dir, pattern_type + '_' + str(min_sup) + '_sup_fsp_'+str(count)+'.json'))

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
                if child.sup[opposite_type] == 0:
                    p.con = -1
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


def translate_result(dir_name):
    data_dir = os.path.join(get_global_val('result_dir'), "{}_9_total".format(dir_name))
    event_dir = os.path.join(get_global_val('data_dir'), dir_name)
    event_id = load_event_id(os.path.join(event_dir, 'event_id.json'))

    for p_type in ['csp', 'fsp']:
        files = os.listdir(data_dir)
        files = list(filter(lambda x: '_sup' in x and p_type in x, files))
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
                if p_type == 'csp':
                    temp.append(d['gr'])
                    temp.append(d['con'])
                res.append(temp)

            file = file.replace("json", "csv")
            if 'pos' in file:
                res = sorted(res, key=lambda x: x[1], reverse=True)
            else:
                res = sorted(res, key=lambda x: x[2], reverse=True)
            with open(os.path.join(data_dir, file), 'w', newline='') as f:
                writer = csv.writer(f)
                if p_type == 'csp':
                    writer.writerow(['seq', 'sup_pos', 'sup_neg', 'growth_rate', 'condition_redundancy'])
                elif p_type == 'fsp':
                    writer.writerow(['seq', 'sup_pos', 'sup_neg'])
                for d in res:
                    writer.writerow(d)


def validate_seq_vector(data_dir, model_idx, use_PCA=False, use_csp=True):
    patterns = []
    if use_csp:
        data = load_json_data(os.path.join(data_dir, 'pos_0.1_sup_csp_'+str(model_idx)+'.json'))
        for i in data:
            # if i['gr'] > 3:
            patterns.append(i['seq'])
        data = load_json_data(os.path.join(data_dir, 'neg_0.1_sup_csp_'+str(model_idx)+'.json'))
        pos_size = len(patterns)
        for i in data:
            # if i['sup']['pos'] == 0:
            #     continue
            patterns.append(i['seq'])
        neg_size = len(patterns)-pos_size
    else:
        data = load_json_data(os.path.join(data_dir, 'pos_0.1_sup_fsp_' + str(model_idx) + '.json'))
        for i in data:
            if i['sup']['neg'] + i['sup']['pos'] >= 0.2:
                patterns.append(i['seq'])
        data = load_json_data(os.path.join(data_dir, 'neg_0.1_sup_fsp_' + str(model_idx) + '.json'))
        pos_size = len(patterns)
        for i in data:
            if i['sup']['neg'] + i['sup']['pos'] >= 0.2:
                if i['seq'] not in patterns:
                    patterns.append(i['seq'])
        neg_size = len(patterns) - pos_size

    print("total patterns: {}, pos:neg = {}:{}".format(len(patterns), pos_size, neg_size))

    # print(patterns)
    y_train = []
    x_train = []
    y_test = []
    x_test = []
    data = load_json_dict(os.path.join(data_dir, 'input_sequences_'+str(model_idx)+'.json'))
    for eff in data:
        for i in data[eff]:
            # i = ["X*M", "M-a", "a+a", "a+a", "a-a", "a+a", "a-a", "a+a", "a+a", "a+a", "a+a"]
            y_train.append(eff)
            _next = build_next(i)
            vec = calculate_seq_vector(_next, patterns)
            x_train.append(vec)
    # test_seq = []
    data = load_json_dict(os.path.join(data_dir, 'test_sequences_'+str(model_idx)+'.json'))

    seq_len = []
    for eff in data:
        for i in data[eff]:
            # test_seq.append(i)
            _next = build_next(i)
            vec = calculate_seq_vector(_next, patterns)
            y_label = eff
            # y_label = 'neutral'
            # for j in vec:
            #     if j == 1:
            #         x_test.append(vec)
            #         y_test.append(eff)
            #         seq_len.append(len(i))
            #         break
            x_test.append(vec)
            seq_len.append(len(i))
            y_test.append(y_label)
    print(len(x_test), len(y_test))

    # for n_c in range(1, 50):
    #     pca = PCA(n_components=n_c)
    #     pca.fit(x_train)
    #     train_x_new = pca.transform(x_train)
    #     print("dimension: {}, explained variance ratio: {}".format(n_c, pca.explained_variance_ratio_.sum()))

    count = 0
    xx_train = []
    yy_train = []
    pos_count = 0
    neg_count = 0
    for i in range(len(x_train)):
        flag = False
        for u in x_train[i]:
            if u == 1:
                count += 1
                flag = True
                xx_train.append(x_train[i])
                yy_train.append(y_train[i])
                break
        if not flag:
            if y_train[i] == 'pos':
                pos_count += 1
            elif y_train[i] == 'neg':
                neg_count += 1
            # y_train[i] = 'neutral'
            # print(y_train[i])
    print("total train samples: {}; hit 0 pattern: {}, pos : neg = {} : {}".format(len(x_train), len(x_train)-count, pos_count, neg_count))

    if use_PCA:
        pca = PCA(n_components=50)
        pca.fit(x_train)
        print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
        train_x_new = pca.transform(x_train)
        clf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight='balanced')
        clf.fit(train_x_new, y_train)
        # if use_csp:
        #     pickle.dump(clf, open(os.path.join(data_dir, 'model_' + str(model_idx) + '.sav'), 'wb'))
        # else:
        #     pickle.dump(clf, open(os.path.join(data_dir, 'model_fsp_' + str(model_idx) + '.sav'), 'wb'))
        test_x_new = pca.transform(x_test)
    else:
        # clf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight='balanced')
        clf = RandomForestClassifier(n_estimators=500, oob_score=True)
        clf.fit(x_train, y_train)
        # if use_csp:
        #     pickle.dump(clf, open(os.path.join(data_dir, 'model_'+str(model_idx)+'.sav'), 'wb'))
        # else:
        #     pickle.dump(clf, open(os.path.join(data_dir, 'model_fsp_' + str(model_idx) + '.sav'), 'wb'))
        test_x_new = x_test

    pred = clf.predict(test_x_new)

    false_seq_len = []
    for i in range(len(seq_len)):
        if pred[i] != y_test[i]:
            false_seq_len.append([seq_len[i], y_test[i]])

    print(confusion_matrix(y_test, pred, labels=['pos', 'neg']))
    print(classification_report(y_test, pred))
    return y_test, pred, false_seq_len
    #
    # seq = load_json_list(data_dir+'all_sequences_symbol_ver.json')
    # seq_id = load_json_list(data_dir+'all_sequences.json')
    #
    # for i in range(len(x_test)):
    #     if y_test[i] != pred[i]:
    #         idx = seq.index(test_seq[i])
    #         print(seq_id[idx]['_id'])
    #         print(y_test[i], pred[i], test_seq[i], x_test[i])



    # test_x_new = pca.transform(x_test)

    # clf = GaussianNB()
    #
    #
    # print(confusion_matrix(y_test, pred, labels=['pos', 'neg']))
    # print(classification_report(y_test, pred))

def calculate_seq_vector(seq, patterns):
    vec = [0]*len(patterns)
    for k in range(len(patterns)):
        p = patterns[k]
        # if p != ["a-a", "a-a"]:
        #     continue
        # print(p)
        flag = True
        idx = 0
        for i in range(len(p)):
            c = p[i]
            # print(c)
            try:
                if idx >= len(seq):
                    flag = False
                    break
                if c in seq[idx]:
                    idx = seq[idx][c] + 1
                else:
                    flag = False
                    break
            except Exception:
                # print(p, idx, seq)
                flag = False  # the pattern is longer than the sequence
                # exit(-1)
        # print(flag)
        if flag:
            # pos_count += d[1]
            vec[k] = 1
    return vec

def recommend_actions(repo_name, model_idx):
    data_dir = os.path.join(get_global_val('result_dir'), repo_name)
    patterns = []
    data = load_json_data(os.path.join(data_dir, 'pos_0.1_sup_csp_'+str(model_idx)+'.json'))
    for i in data:
        patterns.append(i['seq'])
        # if i['sup']['neg'] == 0:
        #     patterns.append(i['seq'])
        # elif i['sup']['pos'] / i['sup']['neg'] > 3:
        #     patterns.append(i['seq'])
    data = load_json_data(os.path.join(data_dir, 'neg_0.1_sup_csp_'+str(model_idx)+'.json'))
    for i in data:
        patterns.append(i['seq'])
        # if i['sup']['pos'] == 0:
        #     patterns.append(i['seq'])
        # elif i['sup']['neg'] / i['sup']['pos'] > 1.7:
        #     if i['seq'] not in patterns:
        #         patterns.append(i['seq'])
    # print(patterns)
    # print(pos_patterns)

    cand = set()
    for i in patterns:
        for j in i:
            cand.add(j)


    # cand1, cand2 = generate_alphabet(patterns, train_seqs)
    # cand = cand1.union(cand2)
    # print(cand1)
    # print(cand2)
    test_ = load_json_dict(os.path.join(data_dir, 'test_sequences_'+str(model_idx)+'.json'))
    test_seqs = []
    for i in test_:
        for q in test_[i]:
            test_seqs.append({'seq': q})

    seq = ['R*X', 'X+E', 'E-H', 'H+Q', 'Q+E', 'E-H', 'H-F', 'F+E', 'E+E', "E-X", "X+U", "U-H", "H+H", "H-H"]
    clf = pickle.load(open(os.path.join(data_dir, 'model_'+str(model_idx)+'.sav'), 'rb'))
    # pred = clf.predict(seq)
    for s in test_seqs:
        s = s['seq']
        print('------------------------------------------------------------')
        print('The full sequence: {}'.format(s))
        new_s = s[0:9]
        subsequent_events = s[9:len(s)]
        do_recommend(new_s, patterns, clf, cand)
        for j in subsequent_events:
            new_s = new_s+[j]
            do_recommend(new_s, patterns, clf, cand)


def generate_alphabet(data1, data2):
    res_1 = set()
    res_2 = set()
    for i in data1:
        for j in i:
            res_1.add(j)

    for i in data2:
        for j in i['seq']:
            res_2.add(j)

    res_2 = res_2-res_1
    return res_1, res_2


def do_recommend(seq, patterns, clf, cand):
    print("The current sequence: {}".format(seq))
    RA = []
    _next = build_next(seq)
    vec = calculate_seq_vector(_next, patterns)
    is_neg = False
    if numpy.any(vec):
        # contain some pattern(s)
        pred_s = clf.predict([vec])[0]
        if pred_s == 'neg':
            is_neg = True
            print(colored("Warning: the trend of this sequence is negative.", 'red'))
        else:
            print(colored("Congrats! The trend of this sequence is positive", 'green'))
    # if not numpy.any(vec) or is_neg:
    tail_e = seq[len(seq)-1][2]
    flag = False
    for i in cand:
        if i[0] == tail_e:
            new_s = seq+[i]
            _next = build_next(new_s)
            vec = calculate_seq_vector(_next, patterns)
            if numpy.any(vec):
                # pred_s = clf.predict([vec])[0]
                pred_s = clf.predict_proba([vec])[0]
                res = {'pos': 0, 'neg': 0}
                for j in range(len(clf.classes_)):
                    res[clf.classes_[j]] = pred_s[j]
                if res['pos'] > res['neg']:
                    RA.append({'action': i, 'prob': res})
                    print(colored("Here we recommend you the following action: {} with the positive probability {}".format(i, res['pos']), 'yellow'))
                    flag = True
    if not flag:
        print('Can not found an appropriate action.')
    return RA

def translate_seq(s, event_map, time_map):
    res = ''
    for m in s:
        res += '['
        for i in m:
            if i in event_map:
                res += event_map[i]
            else:
                res += time_map[i]
            res += ' | '
        res += ']'
    return res


def time_discretize(write_dir, pair):
    data_dir = write_dir
    generate_event_interval(write_dir, pair)

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


def generate_event_interval(write_dir, pair=False):
    data_dir = get_global_val('data_dir')+'sequences/'
    event_interval = {}
    for repo in ['tensorflow']:
        files = list(filter(lambda x: 'issue_sequences' in x and 'cut' in x and repo in x and 'origin' not in x, os.listdir(data_dir)))
        event_interval[repo] = {}
        for file in files:
            data = load_json_list(os.path.join(data_dir, file))
            if 'pos' in file:
                _type = 'pos'
            else:
                _type = 'neg'
            if not pair:
                for d in data:
                    events = d['action_sequence']
                    i = 0
                    while i < len(events):
                        _id = events[i]
                        if _id not in event_interval[repo]:
                            event_interval[repo][_id] = []
                        event_interval[repo][_id].append([events[i+1], _type])
                        i += 2
            else:
                for d in data:
                    events = d['action_sequence']
                    i = 0
                    while i < len(events)-1:
                        try:
                            _id = events[i]+'_'+events[i+2]
                            if _id not in event_interval[repo]:
                                event_interval[repo][_id] = []
                            event_interval[repo][_id].append([events[i + 1], _type])
                            i += 2
                        except Exception:
                            print(i, events)
                            exit(-1)
    write_json_dict(event_interval, os.path.join(write_dir, 'event_interval.json'))


def generate_dataset_event_interval(x_train, y_train, write_dir):
    event_interval = {}
    for i in range(len(x_train)):
        events = x_train[i]
        _type = y_train[i]
        i = 0
        while i < len(events)-1:
            try:
                _id = events[i]+'_'+events[i+2]
                if _id not in event_interval:
                    event_interval[_id] = []
                event_interval[_id].append([events[i + 1], _type])
                i += 2
            except Exception:
                print(i, events)
                exit(-1)
    write_json_dict(event_interval, os.path.join(write_dir, 'event_interval.json'))


def time_discretize_entropy_auto(write_dir, pair, gene_interval=False):
    data_dir = write_dir

    if gene_interval:
        generate_event_interval(write_dir, pair=True)

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


def dataset_time_discretize(x_train, y_train, write_dir):
    generate_dataset_event_interval(x_train, y_train, write_dir)

    event_interval = load_json_dict(os.path.join(write_dir, 'event_interval.json'))
    interval_split = {}
    # information gain

    for e in event_interval:
        data = sorted(event_interval[e], key=lambda x: int(x[0]))
        # if e == 'UnlockedEvent':
        # print(data)
        sum_p, sum_n = calcu_prefix_sum(data)
        p_num = sum_p[len(sum_p) - 1]
        n_num = sum_n[len(sum_n) - 1]
        total_num = len(data)
        H_0 = calculate_entropy(p_num, n_num, total_num)
        split_i = find_split_by_entropy(data, sum_p, sum_n)

        if split_i == -1:
            interval_split[e] = []
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
            # print(e)
            # print(res)
            res = sorted(res, key=lambda x: x[0], reverse=True)
            # print(res)
            interval_split[e] = []
            for pos in res[0][1]:
                interval_split[e].append(data[pos][0])
                # print(interval_split[e])

    write_json_dict(interval_split, os.path.join(write_dir, 'interval_split_auto_entropy.json'))
    # exit(-1)


def dataset_time_discretize_quartile(x_train, y_train, write_dir):
    generate_dataset_event_interval(x_train, y_train, write_dir)

    event_interval = load_json_dict(os.path.join(write_dir, 'event_interval.json'))
    interval_split = {}
    # information gain

    for e in event_interval:
        data = sorted(event_interval[e], key=lambda x: int(x[0]))
        temp = []
        for i in data:
            temp.append(i[0])
        qs = list(numpy.percentile(temp, (25, 50, 75), method='midpoint'))
        interval_split[e] = qs
    write_json_dict(interval_split, os.path.join(write_dir, 'interval_split_auto_entropy.json'))


def generate_dataset(repo_name):
    data_dir = get_global_val('data_dir')+repo_name
    neg_s = load_json_list(os.path.join(data_dir, 'issue_sequences_neg_cut_origin.json'))
    pos_s = load_json_list(os.path.join(data_dir, 'issue_sequences_pos_cut_origin.json'))
    med_s = load_json_list(os.path.join(data_dir, 'issue_sequences_med_cut_origin.json'))
    X = []
    Y = []
    # M = []
    D = []
    for i in neg_s:
        X.append(i['action_sequence'])
        Y.append('neg')
        D.append({'_id': i['_id'], 'seq': i['action_sequence'], 'cls': 'neg'})
    for i in pos_s:
        X.append(i['action_sequence'])
        Y.append('pos')
        D.append({'_id': i['_id'], 'seq': i['action_sequence'], 'cls': 'pos'})
    # for i in med_s:
    #     X.append(i['action_sequence'])
    #     Y.append('med')
    #     D.append({'_id': i['_id'], 'seq': i['action_sequence'], 'cls': 'med'})
    return X, Y, D


def calcu_prefix_sum(data):
    if len(data) == 0:
        return [], []
    pos = [0] * len(data)
    neg = [0] * len(data)
    for i in range(len(data)):
        if data[i][1] == 'pos':
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

