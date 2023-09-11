from preprocess import *
from process_mining import *
from bug_fix_efficiency_classify import *
from contrast_sequential_pattern_mining import *
import matplotlib.pyplot as plt
from sequence_cluster import *
from text_clustering import *
from pre_classification import *

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

    # time_discretize()
    # time_discretize_2_by_entropy()
    # time_discretize_by_entropy()
    # time_discretize_entropy_auto()
    # generate_input_sequence(use_entropy=True)  # for sequence mining
    # show_event_freq()
    show_event_interval()
    # sequence_length_show()
    # cut_sequence(30)
    # calcu_inconsistent_ratio()
    # CDSPM()
    # translate_result()
    # if_event = find_infrequent_event()
    # translate_sequences(['LockedEvent']+list(if_event))
    # load_event_log()
    # get_event_duration()
    # calculate_statics()
    # generate_graph()
