# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from preprocess import *
from bug_fix_efficiency_classify import *
from contrast_sequential_pattern_mining import *
import matplotlib.pyplot as plt
from sequence_cluster import *
from text_clustering import *
from pre_classification import *

if __name__ == '__main__':
    # extract_raw_data(['tensorflow', 'ansible'])
    # select_bug_issue()
    # add_event_to_issues('data/bug_issues.json', 'data/issue_events.json', 'issue')
    # add_event_to_issues('data/prs.json', 'data/pr_events.json', 'pr')
    # select_issue_with_code('data/bug_issues_with_events.json', 'data/prs_with_events.json',
    #                        'data/bug_issues_with_resolutions.json')
    # select_closed_issue('data/bug_issues_with_resolutions.json')
    # add_comment_to_issues('data/closed_bug_issues.json', 'data/issue_comments.json')

    # extract_commit_diff('data/closed_bug_issues.json')
    # generate_commit_diffs()
    # delete_merge_commit('data/commit_diffs.json')
    # limit_commit_filetype('data/commit_diffs_limited.json')
    #
    # add_commitDiff_to_issues('data/closed_bug_issues.json', 'data/commit_diffs_limited.json', 'data/closed_bug_issues.json')

    # issues_loc = calculate_issue_loc('data/closed_bug_issues.json')
    # cluster_by_complexity(issues_loc, 'data/issue_clusters.json')
    #
    # integrate_issue_and_prs('data/closed_bug_issues.json', 'data/prs.json',
    #                         'data/closed_bug_fix.json')
    #
    # generate_bug_fix_features('data/closed_bug_fix.json', 'data/closed_bug_fix_efficiency.json')
    # bug_fix_efficiency = load_json_data('data/closed_bug_fix_efficiency.json')
    # integrate_issue_with_efficiency('data/closed_bug_fix.json', bug_fix_efficiency)
    #
    # generate_sequence_length('data/closed_bug_fix.json')
    # generate_clusters_features('data/closed_bug_fix.json', 'data/issue_clusters.json', 'data/closed_bug_issue_features.json')

    # sequential pattern mining
    # generate_event_id(['data/closed_bug_fix.json'], 'data/event_id.json')
    # generate_input_sequence(['data/clusters_features.csv',
    #                          'data/closed_bug_fix.json',
    #                          'data/event_id.json'],
    #                         'data/closed_bug_fix_sequences.json')

    mining_CSP('data/closed_bug_fix_sequences.json', min_cr=2)
    # for i in range(0, 3):
    #     remove_subsequence_csp(r'data/'+str(i)+'CSP_results.csv', r'data/'+str(i)+'CSP_remained.csv')

    # calculate_fix_time_of_issues('data/closed_bug_issues_with_events.json')
    # quartile = calculate_statistic_of_fix_time('data/closed_bug_issues_with_fix-time.json')
    # select_issues_by_fix_time(quartile, 'data/closed_bug_issues_with_fix-time.json', 'data/efficient_issue.json', 'data/inefficient_issue.json')
    #
    # integrate_issue_and_prs('data/efficient_issue.json', 'data/tensorflow_prs_with_events.json', 'data/efficient_bug_fix.json')
    # integrate_issue_and_prs('data/inefficient_issue.json', 'data/tensorflow_prs_with_events.json',
    #                         'data/inefficient_bug_fix.json')

    # generate_event_id(['data/efficient_bug_fix.json', 'data/inefficient_bug_fix.json'], 'data/event_id.json')
