import copy

from util import *


class Sequence:
    def __init__(self, seq, _next, cls):
        self.seq = seq
        self.next = _next
        self.cls = cls  # pos and neg


class Node:
    def __init__(self, is_leaf=False, val=None, depth=0):
        self.val = val
        self.is_leaf = is_leaf
        self.children = {}
        self.depth = depth
        self.sup = {'pos': 0, 'neg': 0}


class HashTree:
    def __init__(self):
        self.root = Node()

    def insert(self, s):
        root = self.root
        cur_node = root
        for i in range(len(s)):
            if s[i] not in cur_node.children:
                cur_node.children[s[i]] = Node(val=s[i], depth=cur_node.depth + 1)
            cur_node = cur_node.children[s[i]]
            cur_node.is_leaf = False
        cur_node.is_leaf = True

    def print_tree(self):
        node = self.root
        res = [node]
        while len(res) > 0:
            n = res.pop(0)
            children = list(n.children.keys())
            if not n.is_leaf:
                print("element: {}, depth: {}, children: {}, is_leaf: {}".format(n.val, n.depth, children, n.is_leaf))
            else:
                print("element: {}, depth: {}, is_leaf: {}, sup: {}".format(n.val, n.depth, n.is_leaf, n.sup))
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
            # print(node.val, node.is_leaf, node.sup)
            return node.sup
        else:
            raise ValueError('No such node')


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


def generate_input_sequence():
    data_dir = get_global_val('data_dir')

    ignore_events = {'LockedEvent'}

    if not os.path.exists(data_dir + 'event_id.json'):
        generate_event_id(data_dir + 'event_id.json', ignore_events=ignore_events)

    event_id = load_event_id(data_dir + 'event_id.json')

    input_sequences = {}
    for eff in ['low', 'high']:
        input_sequences[eff] = []
        data = load_json_list(data_dir + 'bug_fix_sequences_' + eff + '.json')
        for d in data:
            temp = ''
            occur = None
            for e in d['action_sequence']:
                if e['event_type'] in ignore_events:
                    continue
                if occur is not None:
                    d_t = calculate_delta_t(occur, e['occur_at'])
                    temp += set_duration_symbol(d_t)
                occur = e['occur_at']
                temp += event_id[e['event_type']]
            temp += '='
            i = 0
            cur = []
            while i < len(temp):
                cur.append(temp[i]+temp[i+1])
                i += 2
            input_sequences[eff].append(cur)
    write_json_dict(input_sequences, data_dir+'input_sequences.json')


def set_duration_symbol(t):
    # +: < 7d, -: 7~14d, *:14~28d, .:>=28d
    if t < 7:
        return '+'
    elif t < 14:
        return '-'
    elif t < 28:
        return '*'
    else:
        return '.'


def load_event_id(path):
    with open(path, 'r') as f:
        dic = json.load(f)
        return dic


def temp_test():
    data_dir = get_global_val('data_dir')
    data = load_json_dict(data_dir+'input_sequences.json')
    count = 0
    flag = False
    for s in data['high']:
        for i in range(len(s)):
            if s[i] == 'C':
                for j in range(i+1, len(s)-1):
                    if s[j] == '.':
                        flag = True
        if flag:
            count += 1
            flag = False
    print(count)


def CDSPM():
    # Ck = ['123', '125', '153', '234', '253', '345', '534']
    root_c = HashTree()

    # root_g = None

    data_dir = get_global_val('data_dir')
    data = load_json_dict(data_dir+'input_sequences.json')
    # data = {'low': [['A+', 'B-', 'E+', 'C='], ['A+', 'B-', 'C*', 'D='], ['A+', 'D+', 'B-', 'E+', 'C=']],
    #         'high': [['A*', 'B-', 'C='], ['A*', 'B='], ['A=']]}

    D = []
    Dp_size = len(data['high'])
    Dn_size = len(data['low'])
    print("----------------------  generate next pointers ---------------------")
    for s in data['low']:
        _next = build_next(s)
        seq = Sequence(s, _next, 'neg')
        D.append(seq)
    for s in data['high']:
        _next = build_next(s)
        seq = Sequence(s, _next, 'pos')
        D.append(seq)

    k = 1
    print("positive sequences: {}, negative sequences: {}".format(Dp_size, Dn_size))
    Ck = GSP_ini(D)
    # print(Ck)
    # exit(-1)
    Gk = []
    while len(Ck) > 0:
        print("k={}, generate candidate item sets C_{}: size = {}".format(k, k, len(Ck)))
        for i in Ck:
            root_c.insert(i)
        # root_c.print_tree()
        for s in D:
            count_candidates(root_c.root, s, idx=0, depth=k)

        Fk = get_frequent_pattern(root_c, 0.1, Dn_size, Ck, _type='neg')
        print("number of length = {} fsp: {}".format(k, len(Fk)))
        Gk += get_csp(root_c, 2, Dp_size, Dn_size, Fk)
        print("number of length = {} csp: {}".format(k, len(Gk)))
        # # for conditional discriminative sequential patterns
        # for p in Gk:
        #     # TODO
        #     pass

        Ck = GSP(Fk)
        k += 1
        # print(Fk, len(Ck))
    write_json_data(Gk, data_dir+'csp.json')


def build_next(s):
    _next = [0]*len(s)
    ptr = {}
    for i in range(len(s)-1, -1, -1):
        k = s[i]
        ptr[k] = i
        _next[i] = copy.deepcopy(ptr)
    return _next


def count_candidates(node: Node, s: Sequence, idx, depth):
    if idx >= len(s.seq) or len(node.children) == 0:
        return
    for c in node.children:
        child = node.children[c]
        key = c
        if key in s.next[idx]:
            if child.is_leaf and child.depth == depth:
                child.sup[s.cls] += 1
            skip = s.next[idx][key]
            count_candidates(child, s, skip+1, depth)


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
            if Ck[i][1: len(Ck[i])] == Ck[j][0: len(Ck[j])-1]:
                # print("p1: {}, p2: {}".format(Ck[i], Ck[j]))
                new_item = Ck[i]+[Ck[j][len(Ck[j])-1]]
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
    res = []
    for p in CK:
        sup = root.get_sup(p)
        if (sup[_type]/D_size) >= min_sup:
            # print(p, sup['neg']/D_size)
            res.append(p)
    return res


def get_csp(root: HashTree, min_gr, Dp_size, Dn_size, FK):
    res = []
    for p in FK:
        sup = root.get_sup(p)
        if sup['pos'] == 0:
            # print(p, sup)
            res.append({'seq': p, 'sup': sup, 'gr': -1})
        else:
            gr = (sup['neg']/sup['pos'])*(Dp_size/Dn_size)
            if gr >= min_gr:
                # print(p, sup, gr)
                res.append({'seq': p, 'sup': sup, 'gr': gr})
    return res
