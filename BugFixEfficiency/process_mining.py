from util import *
import pm4py
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random


def translate_sequences(ignore_events):
    data_dir = get_global_val('data_dir')

    for eff in ['high', 'low']:
        data = load_json_list(data_dir + 'bug_fix_sequences_' + eff + '.json')

        with open(data_dir + 'input_traces_' + eff + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trace_id', 'action', 'timestamp'])
            for d in data:
                trace_id = d['_id']
                for a in d['action_sequence']:
                    if a['event_type'] in ignore_events:
                        continue
                    writer.writerow([trace_id, a['event_type'], a['occur_at']])


def find_infrequent_event():
    data_dir = get_global_val('data_dir')
    infrequent_event = { 'high': set(), 'low': set() }
    for eff in ['high', 'low']:
        data = pd.read_csv(data_dir + 'input_traces_' + eff + '.csv')
        event_log = pm4py.format_dataframe(data, case_id='trace_id', activity_key='action', timestamp_key='timestamp')

        activities = pm4py.stats.get_event_attribute_values(event_log, 'concept:name')
        f_activities = list(filter(lambda x: x[1] < 100, activities.items()))
        infrequent_event[eff] = set([x[0] for x in f_activities])

    res = infrequent_event['high'].intersection(infrequent_event['low'])
    return res


def get_event_duration():
    data_dir = get_global_val('data_dir')
    for eff in ['high', 'low']:
        event_duration = { }
        data = load_json_list(data_dir + 'bug_fix_sequences_' + eff + '.json')
        for d in data:
            events = d['action_sequence']
            for i in range(len(events) - 1):
                delta_t = calculate_delta_t(events[i]['occur_at'], events[i + 1]['occur_at'])
                _id = events[i]['event_type'] + '-' + events[i + 1]['event_type']
                if _id not in event_duration:
                    event_duration[_id] = []
                event_duration[_id].append(delta_t)
        write_json_dict(event_duration, data_dir + 'event_duration_' + eff + '.json')


def calculate_statics():
    data_dir = get_global_val('data_dir')
    data = { 'high': { }, 'low': { } }
    for eff in ['high', 'low']:
        data[eff] = load_json_dict(data_dir + 'event_duration_' + eff + '.json')
        # calculate statics of event translation
        statics = { }
        for i in data[eff]:
            statics[i] = [numpy.mean(data[eff][i]), numpy.median(data[eff][i]), numpy.std(data[eff][i])]
            statics[i].append(statics[i][0] / statics[i][2])
            statics[i].append(len(data[eff][i]))
            # print(i, data[i])
        write_json_dict(statics, data_dir + 'duration_statics_' + eff + '.json')

        with open(data_dir + 'duration_statics_' + eff + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['event_translation', 'mean', 'median', 'std', 'cv', 'freq'])
            for i in statics:
                writer.writerow([i] + statics[i])

    # # normalize
    # total = {}
    # for eff in ['high', 'low']:
    #     for i in data[eff]:
    #         if i not in total:
    #             total[i] = []
    #         total[i] += data[eff][i]
    # total_statics = {}
    # for i in total:
    #     total_statics[i] = {'mean': numpy.mean(total[i]), 'std': numpy.std(total[i])}
    #
    # for eff in ['high', 'low']:
    #     for i in data[eff]:
    #         for j in range(len(data[eff][i])):
    #             if total_statics[i]['std'] == 0:
    #                 data[eff][i][j] = 0
    #             else:
    #                 data[eff][i][j] = (data[eff][i][j] - total_statics[i]['mean'])/total_statics[i]['std']


def load_event_log():
    data_dir = get_global_val('data_dir')
    figure_dir = get_global_val('figure_dir')

    for eff in ['high', 'low']:
        data = pd.read_csv(data_dir + 'input_traces_' + eff + '.csv')
        # num_events = len(data)
        # num_cases = len(data.trace_id.unique())
        # print("Number of events: {}\nNumber of cases: {}".format(num_events, num_cases))
        event_log = pm4py.format_dataframe(data, case_id='trace_id', activity_key='action', timestamp_key='timestamp')

        dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
        print("start: {}\n end: {}\n".format(start_activities, end_activities))
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, figure_dir + 'pm/' + eff + '.png')

        # dfg_s = dict(sorted(dfg.items(), key=lambda x: x[1]))
        # activities = pm4py.stats.get_event_attribute_values(event_log, 'concept:name')
        # f_activities = list(filter(lambda x: x[1] < 100, activities.items()))
        # print("{} \n".format(f_activities))
        # for a in activities:
        #     self_d = pm4py.stats.get_activity_position_summary(event_log, activity=a)
        #     print("{} in positions: {}".format(a, self_d))

        dfg_f = dict(list(filter(lambda x: x[1] < 100, dfg.items())))
        f_relations = list(dfg_f.keys())
        print(len(dfg), len(f_relations))
        #
        # # f_relations = list(filter(lambda x: 'LockedEvent' in x, dfg.keys()))
        # # print(f_relations)
        filtered_log = pm4py.filter_directly_follows_relation(event_log, relations=f_relations, retain=False)
        dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_log)
        # print("start: {}\n end: {}\n".format(start_activities, end_activities))
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, figure_dir + 'pm/' + eff + '_filtered.png')

        # filtered_log = pm4py.filtering.filter_variants_by_coverage_percentage(event_log, min_coverage_percentage=0.05)

        # filtered_log = pm4py.filtering.filter_variants_top_k(event_log, k=10)
        # dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_log)
        # pm4py.save_vis_dfg(dfg, start_activities, end_activities, figure_dir + 'pm/' + eff + '_top10.png')
        # print(dfg.items())

        # net, im, fm = pm4py.discover_petri_net_alpha(event_log)
        # pm4py.save_vis_petri_net(net, im, fm, figure_dir+'pm/petri_'+eff+'.png')


def generate_graph():
    data_dir = get_global_val('data_dir')
    figure_dir = get_global_val('figure_dir')

    data = {}
    event_set = set()
    for eff in ['high', 'low']:
        data[eff] = load_json_list(data_dir+'duration_statics_'+eff+'.json')
        for i in data[eff]:
            events = i['_id'].split('-')
            event_set.add(events[0])
            event_set.add(events[1])
    if not os.path.exists(data_dir+'event_id.json'):
        generate_event_id(event_set, data_dir+'event_id.json')

    event_id = load_json_data(data_dir+'event_id.json')

    for eff in ['high', 'low']:
        graph_data = []
        for i in data[eff]:
            (u, v) = i['_id'].split('-')
            u = event_id[u]
            v = event_id[v]
            # d = i['data'][4]  # frequency
            d = i['data'][0]  # mean fix time
            if i['data'][4] < 1000:
                continue
            graph_data.append([u, v, d])
            print(u, v, d)


        # m = normalize([x[2] for x in graph_data])
        # for i in range(len(graph_data)):
        #     graph_data[i][2] = m[i]

        # graph_data = sorted(graph_data, key=lambda x: x[2], reverse=True)
        weights = [x[2] for x in graph_data]
        weights = sorted(weights, reverse=True)
        weight_map = {}
        count = 0
        for i in weights:
            if i not in weight_map:
                weight_map[i] = count
                count += 1
        print(weights)

        c_map = cm.get_cmap('Spectral')
        # rgba = cmap(0.5)
        # print(rgba)

        G = nx.DiGraph()
        for u, v, d in graph_data:
            G.add_edge(u, v, weight=d)
        print("total nodes: {}, total edges: {}".format(G.number_of_nodes(), G.number_of_edges()))
        # pos = nx.spring_layout(G, iterations=20)
        # pos = nx.circular_layout(G)
        pos = nx.nx_pydot.pydot_layout(G, prog='dot')
        # pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        label = { }
        for (u, v, d) in G.edges(data=True):
            label[(u, v)] = str(d['weight'])

        print(G.edges(data=True))
        edge_color = [c_map(weight_map[d['weight']]/len(weight_map)) for (u, v, d) in G.edges(data=True)]

        # print(edge_color)
        nx.draw_networkx_edges(G, pos, edge_color=edge_color)
        nx.draw_networkx_nodes(G, pos, node_color='#FFF5EE', edgecolors='#000000')
        nx.draw_networkx_labels(G, pos, font_color='#000000')
        # nx.draw_networkx_edge_labels(G, pos, label)

        plt.savefig(figure_dir+eff+'_fix_time.eps', dpi=600, format='eps')
        plt.show()
