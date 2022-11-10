import os
import h5py
import math
import pandas as pd
import numpy as np
from geopy.distance import geodesic

dataset = "nyc-taxi"
date_range = ["4/1/2016 00:00:00", "6/30/2016 23:59:59"]
delta = np.timedelta64(30, "m")
freq = "30min"
normalized_k = .1


# Generated file is not aligned. Please refer to gen_h5_chicago.py
def get_adjacency_matrix(position_dict, normalized_k):
    # position_dict (lat, lon)
    num_sensors = len(position_dict)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=float)
    dist_mx[:] = np.inf

    for i in range(num_sensors):
        for j in range(num_sensors):
            dist = geodesic(position_dict[i], position_dict[j]).km
            dist_mx[i, j] = dist_mx[j, i] = dist

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx


if __name__ == '__main__':
    init_time = pd.to_datetime(date_range[0], format='%m/%d/%Y %H:%M:%S')
    end_time = pd.to_datetime(date_range[1], format='%m/%d/%Y %H:%M:%S')
    num_time = math.ceil((end_time - init_time) / delta)
    print(dataset)

    if dataset == 'nyc-bike':
        csv = ["201604-citibike-tripdata.csv", "201604-citibike-tripdata.csv", "201606-citibike-tripdata.csv"]
        df = pd.concat([pd.read_csv(f"./original/nyc-bike/{_csv}") for _csv in csv])
        df['starttime'] = pd.to_datetime(df["starttime"], format='%m/%d/%Y %H:%M:%S')
        df['stoptime'] = pd.to_datetime(df["stoptime"], format='%m/%d/%Y %H:%M:%S')
        df.dropna(inplace=True)
        pick, drop = df[['starttime', 'start station id']], df[['stoptime', 'end station id']]

        df_pick = df[["start station id", "start station latitude", "start station longitude"]]
        df_drop = df[["end station id", "end station latitude", "end station longitude"]]
        df_pick.columns = ["id", "latitude", "longitude"]
        df_drop.columns = ["id", "latitude", "longitude"]
        df = pd.concat([df_pick, df_drop]).groupby('id', as_index=False)
        df = df.mean()  # may have a little offset, ignore them
        position_dict = {row.id: (row.latitude, row.longitude) for row in df.itertuples()}

    elif dataset == 'nyc-taxi':
        # since original data is too huge, extract records only in 2016Q2
        if not os.path.exists(f"./original/nyc-taxi/2016_Yellow_Taxi_Trip_Data_Q2.csv"):
            df_iter = pd.read_csv(f"./original/nyc-taxi/2016_Yellow_Taxi_Trip_Data.csv", chunksize=10000)

            data = pd.DataFrame()
            for df in df_iter:
                print(f"{pd.to_datetime(df['tpep_pickup_datetime'].iloc[0])}")
                df['tpep_pickup_datetime'] = pd.to_datetime(df["tpep_pickup_datetime"])
                df['tpep_dropoff_datetime'] = pd.to_datetime(df["tpep_dropoff_datetime"])
                df = df[df['tpep_pickup_datetime'].apply(lambda r: r.year == 2016 and r.month in [4, 5, 6])]
                df = df[df['tpep_dropoff_datetime'].apply(lambda r: r.year == 2016 and r.month in [4, 5, 6])]
                data = pd.concat([data, df])
            data.to_csv(f"./original/nyc-taxi/2016_Yellow_Taxi_Trip_Data_Q2.csv")

        df = pd.read_csv(f"./original/nyc-taxi/2016_Yellow_Taxi_Trip_Data_Q2.csv")
        df['tpep_pickup_datetime'] = pd.to_datetime(df["tpep_pickup_datetime"])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df["tpep_dropoff_datetime"])
        df.dropna(inplace=True)
        pick, drop = df[['tpep_pickup_datetime', 'PULocationID']], df[['tpep_dropoff_datetime', 'DOLocationID']]

        df_pick = df[["PULocationID", "pickup_latitude", "pickup_longitude"]]
        df_drop = df[["DOLocationID", "dropoff_latitude", "dropoff_longitude"]]
        df_pick.columns = ["id", "latitude", "longitude"]
        df_drop.columns = ["id", "latitude", "longitude"]
        df = pd.concat([df_pick, df_drop]).groupby('id', as_index=False)
        df = df.mean()  # may have a little offset, ignore them
        position_dict = {row.id: (row.latitude, row.longitude) for row in df.itertuples()}

    elif dataset == 'chicago-bike':
        csv = ["Divvy_Trips_2016_04.csv", "Divvy_Trips_2016_05.csv", "Divvy_Trips_2016_06.csv"]
        df = pd.concat([pd.read_csv(f"./original/chicago-bike/{_csv}") for _csv in csv])
        df['starttime'] = pd.to_datetime(df["starttime"], format='%m/%d/%Y %H:%M')
        df['stoptime'] = pd.to_datetime(df["stoptime"], format='%m/%d/%Y %H:%M')
        df.dropna(inplace=True)
        pick, drop = df[['starttime', 'from_station_id']], df[['stoptime', 'to_station_id']]

        df = pd.read_csv(f"./original/chicago-bike/Divvy_Stations_2016_Q1Q2.csv")
        df = df[["id", "latitude", "longitude"]]
        position_dict = {row.id: (row.latitude, row.longitude) for row in df.itertuples()}

    elif dataset == 'chicago-taxi':
        # since original data is too huge, extract records only in 2016Q2
        if not os.path.exists(f"./original/chicago-taxi/Taxi_Trips_2016Q2.csv"):
            df_iter = pd.read_csv(f"./original/chicago-taxi/Taxi_Trips.csv", chunksize=10000)

            data = pd.DataFrame()
            for df in df_iter:
                # print(f"{pd.to_datetime(df['Trip Start Timestamp'].iloc[0])}")
                df['Trip Start Timestamp'] = pd.to_datetime(df["Trip Start Timestamp"])
                df = df[df['Trip Start Timestamp'].apply(lambda r: r.year == 2016)]
                df = df[df['Trip End Timestamp'].apply(lambda r: r.year == 2016)]
                data = pd.concat([data, df])
                print(len(df))
            data.to_csv(f"./original/chicago-taxi/Taxi_Trips_2016Q2.csv")

        df = pd.read_csv(f"./original/chicago-taxi/Taxi_Trips_2016Q2.csv")
        df['Trip Start Timestamp'] = pd.to_datetime(df["Trip Start Timestamp"])
        df['Trip End Timestamp'] = pd.to_datetime(df["Trip End Timestamp"])
        df.dropna(inplace=True)
        pick, drop = df[['Trip Start Timestamp', 'Pickup Community Area']], \
                     df[['Trip End Timestamp', 'Dropoff Community Area']]

        df_pick = df[["Pickup Community Area", "Pickup Centroid Latitude", "Pickup Centroid Longitude"]]
        df_drop = df[["Dropoff Community Area", "Dropoff Centroid Latitude", "Dropoff Centroid Longitude"]]
        df_pick.columns = ["id", "latitude", "longitude"]
        df_drop.columns = ["id", "latitude", "longitude"]
        df = pd.concat([df_pick, df_drop]).groupby('id', as_index=False)
        df = df.mean()  # may have a little offset, ignore them
        position_dict = {row.id: (row.latitude, row.longitude) for row in df.itertuples()}

    else:
        raise ValueError()

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

    position_dict = {station_mapping[s]: p for s, p in position_dict.items() if s in station_mapping.keys()}
    adj_mx = get_adjacency_matrix(position_dict, normalized_k)

    # to df and set index
    time = np.array(pd.date_range(init_time, end_time, freq=freq).strftime('%Y-%m-%d %H:%M:%S'))
    data = np.stack([pick_np, drop_np], -1)
    f = h5py.File(f"./h5data/{dataset}.h5", 'w')
    f.create_dataset("raw_data", data=data)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    print("end")
