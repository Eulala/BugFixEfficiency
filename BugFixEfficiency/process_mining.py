from util import *
import pm4py



def translate_sequences():
    data_dir = get_global_val('data_dir')

    for eff in ['high', 'low']:
        data = load_json_list(data_dir+'bug_fix_sequences_'+eff+'.json')

        with open(data_dir+'input_traces_'+eff+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trace_id', 'action', 'timestamp'])
            for d in data:
                trace_id = d['_id']
                for a in d['action_sequence']:
                    writer.writerow([trace_id, a['event_type'], a['occur_at']])


def load_event_log():
    data_dir = get_global_val('data_dir')
    figure_dir = get_global_val('figure_dir')

    for eff in ['high', 'low']:
        data = pd.read_csv(data_dir + 'input_traces_'+eff+'.csv')
        # num_events = len(data)
        # num_cases = len(data.trace_id.unique())
        # print("Number of events: {}\nNumber of cases: {}".format(num_events, num_cases))
        event_log = pm4py.format_dataframe(data, case_id='trace_id', activity_key='action', timestamp_key='timestamp')
        start_activities = pm4py.get_start_activities(event_log)
        end_activities = pm4py.get_end_activities(event_log)
        print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))


        dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, figure_dir+'pm/'+eff+'.png')