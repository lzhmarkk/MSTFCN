import os
import math
import pandas as pd
import multiprocessing
from collections import defaultdict

n_threads = 16


def map(p_id, dir, csv_list, queue):
    years = {"R": defaultdict(int), "B": defaultdict(int)}
    for _, f in enumerate(csv_list):
        print(f"p{p_id}\t[{_ + 1}/{len(csv_list)}]\t{f}")

        try:
            df = pd.read_csv(os.path.join(dir, f),
                             usecols=[2, 4, 6, 7, 9, 11, 12], encoding_errors='ignore', on_bad_lines='skip')
            df.columns = ["mode", "starttime", "start_lat", "start_lon", "stoptime", "stop_lat", "stop_lon"]
            df.dropna(inplace=True)
            modes = df['mode']
            times = pd.to_datetime(df['starttime'])
            for m, t in zip(modes, times):
                years[m][(int(t.year), int(t.month))] += 1
        except:
            print(f"ERROR"+"-"*50)
            print(f"p{p_id}\t[{_ + 1}/{len(csv_list)}]\t{f}")
            print(f"ERROR"+"-"*50)

    print(f"p{p_id} finished")
    queue.put(years)


def reduce(years_list):
    years = {"R": defaultdict(int), "B": defaultdict(int)}
    for e in years_list:
        for mode in e:
            for k, v in e[mode].items():
                years[mode][k] += v

    for mode in years:
        _ = {'R': 'railway', 'B': 'bus'}[mode]
        with open(f"./beijing-{_}-distribute.txt", 'w') as f:
            for year in range(2010, 2023):
                for season in range(0, 4):
                    sum = years[mode][(year, 4 * season)] + years[mode][(year, 4 * season + 1)] + \
                          years[mode][(year, 4 * season + 2)]
                    if sum:
                        print(f"{year} Q{season + 1} {sum}")
                        f.write(f"{year} Q{season + 1} {sum}\r\n")


if __name__ == '__main__':
    years = {"R": defaultdict(int), "B": defaultdict(int)}
    dir = "../original/Beijing_IC_record"
    csv = os.listdir(dir)
    csv = list(filter(lambda c: os.path.getsize(os.path.join(dir, c)) > 0 and (c[0] == 't' and 'csv' in c), csv))
    bs = math.ceil(len(csv) / n_threads)

    queue = multiprocessing.Queue()
    jobs = []
    for p_id in range(n_threads):
        csvs = csv[p_id * bs:(p_id + 1) * bs]
        p = multiprocessing.Process(name=f'p{p_id}', target=map, args=(p_id, dir, csvs, queue))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = [queue.get() for j in jobs]
    reduce(results)
