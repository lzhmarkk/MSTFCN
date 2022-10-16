import h5py
import pandas as pd
import numpy as np
import math
import os

dataset = "chicago-taxi"

if __name__ == '__main__':
    date_range = ["4/1/2016 00:00:00", "6/30/2016 23:59:59"]
    init_time = pd.to_datetime(date_range[0], format='%m/%d/%Y %H:%M:%S')
    end_time = pd.to_datetime(date_range[1], format='%m/%d/%Y %H:%M:%S')
    delta = np.timedelta64(30, "m")
    num_time = math.ceil((end_time - init_time) / delta)

    if dataset == 'chicago-bike':
        csv = ["Divvy_Trips_2016_04.csv", "Divvy_Trips_2016_05.csv", "Divvy_Trips_2016_06.csv"]

        df = pd.concat([pd.read_csv(f"./original/Chicago-bike/{_csv}") for _csv in csv])
        df['starttime'] = pd.to_datetime(df["starttime"], format='%m/%d/%Y %H:%M')
        df['stoptime'] = pd.to_datetime(df["stoptime"], format='%m/%d/%Y %H:%M')
        pick, drop = df[['starttime', 'from_station_id']], df[['stoptime', 'to_station_id']]
        pick = pick[(init_time <= pick['starttime']) & (pick['starttime'] <= end_time)]
        drop = drop[(init_time <= drop['stoptime']) & (drop['stoptime'] <= end_time)]

        stations = set()
        stations = stations.union(list(pick['from_station_id']))
        stations = stations.union(list(drop['to_station_id']))
        stations = list(stations)
        stations.sort()
        num_stations = len(stations)
        station_mapping = {s: i for i, s in zip(range(num_stations), stations)}

        pick_np = np.zeros((num_time, num_stations), dtype=int)
        drop_np = np.zeros((num_time, num_stations), dtype=int)

        for row in pick.itertuples():
            start_bin = int((pd.to_datetime(row.starttime) - init_time) / delta)
            start_station = station_mapping[row.from_station_id]
            pick_np[start_bin, start_station] += 1
        for row in drop.itertuples():
            end_bin = int((pd.to_datetime(row.stoptime) - init_time) / delta)
            end_station = station_mapping[row.to_station_id]
            drop_np[end_bin, end_station] += 1

        # to df and set index
        time = np.array(pd.date_range(init_time, end_time, freq="30min").strftime('%Y-%m-%d %H:%M:%S'))
        data = np.stack([pick_np, drop_np], -1)
        f = h5py.File(f"./h5data/chicago-bike.h5", 'w')
        f.create_dataset("raw_data", data=data)
        f.create_dataset("time", data=time)

    elif dataset == 'chicago-taxi':
        if not os.path.exists(f"./original/Chicago-taxi/Taxi_Trips_2016.csv"):
            df_iter = pd.read_csv(f"./original/Chicago-taxi/Taxi_Trips.csv", chunksize=10000)

            data = pd.DataFrame()
            for df in df_iter:
                print(f"{pd.to_datetime(df['Trip Start Timestamp'].iloc[0])}")
                df['Trip Start Timestamp'] = pd.to_datetime(df["Trip Start Timestamp"])
                df = df[df['Trip Start Timestamp'].apply(lambda r: r.year == 2016)]
                df.dropna(inplace=True)
                data = pd.concat([data, df])
            data.to_csv(f"./original/Chicago-taxi/Taxi_Trips_2016.csv")

        df = pd.read_csv(f"./original/Chicago-taxi/Taxi_Trips_2016.csv")
        df['Trip Start Timestamp'] = pd.to_datetime(df["Trip Start Timestamp"])
        df['Trip End Timestamp'] = pd.to_datetime(df["Trip End Timestamp"])
        pick, drop = df[['Trip Start Timestamp', 'Pickup Community Area']], df[['Trip End Timestamp', 'Dropoff Community Area']]
        pick.columns = ['starttime', 'from_station_id']
        drop.columns = ['stoptime', 'to_station_id']
        pick = pick[(init_time <= pick['starttime']) & (pick['starttime'] <= end_time)]
        drop = drop[(init_time <= drop['stoptime']) & (drop['stoptime'] <= end_time)]

        stations = set()
        stations = stations.union(list(pick['from_station_id']))
        stations = stations.union(list(drop['to_station_id']))
        stations = list(stations)
        stations.sort()
        num_stations = len(stations)
        station_mapping = {s: i for i, s in zip(range(num_stations), stations)}

        pick_np = np.zeros((num_time, num_stations), dtype=int)
        drop_np = np.zeros((num_time, num_stations), dtype=int)

        for row in pick.itertuples():
            start_bin = int((pd.to_datetime(row.starttime) - init_time) / delta)
            start_station = station_mapping[row.from_station_id]
            pick_np[start_bin, start_station] += 1
        for row in drop.itertuples():
            end_bin = int((pd.to_datetime(row.stoptime) - init_time) / delta)
            end_station = station_mapping[row.to_station_id]
            drop_np[end_bin, end_station] += 1

        # to df and set index
        time = np.array(pd.date_range(init_time, end_time, freq="30min").strftime('%Y-%m-%d %H:%M:%S'))

        f = h5py.File(f"./h5data/chicago-taxi.h5", 'w')
        data = np.stack([pick_np, drop_np], -1)
        f.create_dataset("raw_data", data=data)
        f.create_dataset("time", data=time)
        # todo adj_mx
    else:
        raise ValueError()
