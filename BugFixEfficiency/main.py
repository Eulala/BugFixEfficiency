import os.path

from preprocess import *
from process_mining import *
from bug_fix_efficiency_classify import *
from contrast_sequential_pattern_mining import *
import matplotlib.pyplot as plt
from sequence_cluster import *
from text_clustering import *
from pre_classification import *


def conduct_CDSPM():
    classify_sequence_by_avgtime(min_len=20)
    # model_sequence()

    # data_dir = get_global_val('result_dir') + 'event_time/len20/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # cut_sequence(20, interval=2)
    # time_discretize(data_dir, suffix='20_2', pair=False)
    # time_discretize_entropy_auto(data_dir, suffix='20_2', pair=False)
    # data_dir = get_global_val('result_dir') + 'event_time/len20/quartile/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # generate_input_sequence(data_dir, use_entropy=False, file_suffix='_20_2')  # for sequence mining
    # data_dir = get_global_val('result_dir') + 'event_time/len20/entropy/'
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # generate_input_sequence(data_dir, use_entropy=True, file_suffix='_20_2')
    # data_dir = get_global_val('result_dir') + 'event_time/len20/quartile/'
    # CDSPM(data_dir)
    # data_dir = get_global_val('result_dir') + 'event_time/len20/entropy/'
    # CDSPM(data_dir)

    data_dir = get_global_val('result_dir') + 'event_time_event/len20/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cut_sequence(20, interval=3)
    time_discretize(data_dir, suffix='20_3', pair=True)
    time_discretize_entropy_auto(data_dir, suffix='20_3', pair=True)
    data_dir = get_global_val('result_dir') + 'event_time_event/len20/quartile/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    generate_input_sequence_ete(data_dir, use_entropy=False, file_suffix='_20_3')  # for sequence mining
    data_dir = get_global_val('result_dir') + 'event_time_event/len20/entropy/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    generate_input_sequence_ete(data_dir, use_entropy=True, file_suffix='_20_3')
    data_dir = get_global_val('result_dir') + 'event_time_event/len20/quartile/'
    CDSPM(data_dir)
    data_dir = get_global_val('result_dir') + 'event_time_event/len20/entropy/'
    CDSPM(data_dir)


if __name__ == '__main__':
    initialize()

    # extract_raw_data()
    # find_commit_repo()
    # commits = get_commit_list()
    # select_commits(commits)
    # select_closed_issue()

    # generate_commit_loc()
    # add_commitDiff_to_issues()

    # modify_pr_occur()
    # calculate_fix_time()
    # normalize_fix_time()
    # set_efficiency()
    # generate_sequence()
    # translate_sequences(['LockedEvent'])  # for process mining

    issue_preprocess()


    # conduct_CDSPM()
    # translate_result()

    calcu_inconsistent_ratio()

    # show_event_freq()
    # show_event_interval()
    # sequence_length_show()


    # if_event = find_infrequent_event()
    # translate_sequences(['LockedEvent']+list(if_event))
    # load_event_log()
    # get_event_duration()
    # calculate_statics()
    # generate_graph()
