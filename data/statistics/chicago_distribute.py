import pandas as pd
from collections import defaultdict
import os


def taxi():
    """
    2016 Q1 5078886
    2016 Q2 8555370
    2016 Q3 7888726
    2016 Q4 2313611
    """
    csv = f"./original/chicago-taxi/Taxi_Trips_-_2016.csv"
    df_iter = pd.read_csv(csv, chunksize=10000)

    years = defaultdict(int)
    for df in df_iter:
        times = df['Trip Start Timestamp'].values
        for t in times:
            y = t[6:10]
            m = t[0:2]
            years[(int(y), int(m))] += 1

    f = open(f"./chicago-taxi-distribute.txt", 'w')
    for year in range(2013, 2023):
        for season in range(0, 4):
            sum = years[(year, 4 * season)] + years[(year, 4 * season + 1)] + years[(year, 4 * season + 2)]
            if sum:
                print(f"{year} Q{season + 1} {sum}")
                f.write(f"{year} Q{season + 1} {sum}\r\n")
    f.close()


def bike():
    """
    2014 Q3 937943
    2015 Q2 893890
    2015 Q3 1238636
    2016 Q2 1072827
    2016 Q3 1273912
    2017 Q2 1119814
    2017 Q3 1397232
    2018 Q2 1059681
    2018 Q3 1313807
    2019 Q2 1108163
    2019 Q3 1455189
    2020 Q3 1543972
    2021 Q2 1598458
    2021 Q3 2191725
    2022 Q2 1775311
    2022 Q3 1487271
    """
    years = defaultdict(int)
    dir = "./original/chicago-bike"
    csv = os.listdir(dir)
    for _, f in enumerate(csv):
        if not ('trip' in f or 'Trip' in f):
            continue
        print(f"[{_ + 1}/{len(csv)}] {f}")

        df = pd.read_csv(os.path.join(dir, f))
        if 'starttime' in df.columns:
            times = pd.to_datetime(df['starttime'])
        elif 'started_at' in df.columns:
            times = pd.to_datetime(df['started_at'])
        elif 'start_time' in df.columns:
            times = pd.to_datetime(df['start_time'])
        elif '01 - Rental Details Local Start Time' in df.columns:
            times = pd.to_datetime(df['01 - Rental Details Local Start Time'])
        else:
            print(df.columns)
            raise ValueError()
        for t in times:
            years[(int(t.year), int(t.month))] += 1

    f = open(f"./chicago-bike-distribute.txt", 'w')
    for year in range(2013, 2023):
        for season in range(0, 4):
            sum = years[(year, 4 * season)] + years[(year, 4 * season + 1)] + years[(year, 4 * season + 2)]
            if sum:
                print(f"{year} Q{season + 1} {sum}")
                f.write(f"{year} Q{season + 1} {sum}\r\n")
    f.close()


if __name__ == '__main__':
    taxi()
    # bike()
