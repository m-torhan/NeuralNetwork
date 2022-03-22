#!/usr/bin/env python3

import git
import os
import  matplotlib.pyplot as plt

perf_report_path = r'./performance_report/'
perf_test_path = r'../tests/performance_tests/Build/Linux/Release/NeuralNetwork_performance_tests'
perf_test_tmp_result_path = perf_report_path + r'tmp_result.txt'
perf_test_results_path = perf_report_path + r'results/'
perf_test_plots_path = perf_report_path + r'plots/'

plt.style.use('dark_background')

os.system(r'./build_release_sse.sh')

repo = git.Repo('./..')

repo_diff = repo.index.diff(None)
if len(repo_diff) > 0:
    print('Please commit your changes before running this script. Uncommited changes found in files:')
    for diff in repo_diff:
        print(f'  {diff.a_path}')
    exit(1)
    
commit_hash = repo.head.commit.hexsha
print(f'Commit hash = {commit_hash}')

if not(os.path.isdir(perf_report_path)):
    os.mkdir(perf_report_path)

os.system(f'{perf_test_path} > {perf_test_tmp_result_path}')

print('Gathering test results.')

results = {}
with open(perf_test_tmp_result_path, 'r') as tmp_result_file:
    for line in tmp_result_file:
        line_split = line.split()
        if line_split[0][:3] == 'BM_':
            results[line_split[0]] = f'{line_split[1]} {line_split[2]}'

os.remove(perf_test_tmp_result_path)

if not(os.path.isdir(perf_test_results_path)):
    os.mkdir(perf_test_results_path)

for key, value in results.items():
    print(f'  test: {key}')
    with open(os.path.join(perf_test_results_path, key + '.txt'), 'a+') as result_file:
        result_file.write(f'{commit_hash} {value}\n')

all_tests = []

for _, _, fnames in os.walk(perf_test_results_path):
    for fname in fnames:
        all_tests.append(fname.split('.')[0])
    break

if not(os.path.isdir(perf_test_plots_path)):
    os.mkdir(perf_test_plots_path)

all_commits = [c.hexsha for c in repo.iter_commits()]

print('Preparing plots.')
for test in all_tests:
    results = {c: None for c in all_commits[::-1]}
    with open(os.path.join(perf_test_results_path, test + '.txt')) as result_file:
        for line in result_file:
            commit, time, _ = line.split()
            results[commit] = int(time)
    
    x = []
    y = []
    for commit, time in results.items():
        if time is not None:
            x.append(commit[:6])
            y.append(time)
    
    fig = plt.figure(figsize=(12, 3), facecolor=(13./255, 17./255, 23./255))
    ax = plt.gca()
    ax.set_facecolor((13./255, 17./255, 23./255))

    plt.plot(x, y, color='aqua')

    plt.xticks(rotation=90)
    plt.ylim(0, None)
    plt.title(test)
    plt.grid('both')
    plt.savefig(os.path.join(perf_test_plots_path, test + '.png'), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.clf()

remote_url = repo.remotes.origin.url

print(remote_url)

print('Creating report.')
with open(os.path.join(perf_report_path, 'report.md'), 'w') as report_file:
    for test in all_tests:
        report_file.write(f'![{test}]({remote_url}/blob/master/Build/{perf_test_plots_path}/{test}.png)\n')

print('All done.')
