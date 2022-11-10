import os
import math
import h5py
import shapefile
import numpy as np
import pandas as pd
from tqdm import tqdm
import shapely.geometry as geometry
from pyproj import CRS, Transformer
from gen_h5 import get_adjacency_matrix

date_range = ["4/1/2016 00:00:00", "6/30/2016 23:59:59"]
delta = np.timedelta64(30, "m")
freq = "30min"
normalized_k = .1
mini_batch = False


def read_geo(proj_file_path, shape_file_path):
    with open(proj_file_path) as prj_f:
        prj = prj_f.read()
    world_crs = CRS.from_epsg(4326)
    chicago_crs = CRS.from_string(prj)
    gps_proj = Transformer.from_crs(world_crs, chicago_crs)
    sf = shapefile.Reader(shape_file_path)
    regions = [geometry.shape(s) for s in sf.shapes()]

    def gps2region(lat, lon):
        p = gps_proj.transform(lat, lon)  # world to chicago
        p = geometry.Point(p)
        for i, region in enumerate(regions):
            if region.contains(p):
                return i
        return -1

    return gps2region, regions


if __name__ == '__main__':
    init_time = pd.to_datetime(date_range[0], format='%m/%d/%Y %H:%M:%S')
    end_time = pd.to_datetime(date_range[1], format='%m/%d/%Y %H:%M:%S')
    num_time = math.ceil((end_time - init_time) / delta)

    gps2region, regions = read_geo(
        proj_file_path='./original/chicago-taxi/CensusTracts/geo_export_85989bc8-888f-407b-824d-a2a7bab9cdeb.prj',
        shape_file_path='./original/chicago-taxi/CensusTracts/geo_export_85989bc8-888f-407b-824d-a2a7bab9cdeb.shp'
    )

    flows = {'bike': np.zeros((num_time, len(regions), 2), dtype=int),
             'taxi': np.zeros((num_time, len(regions), 2), dtype=int)}

    for dataset in ['taxi', 'bike']:
        if dataset == 'bike':
            csv = ["Divvy_Trips_2016_04.csv", "Divvy_Trips_2016_05.csv", "Divvy_Trips_2016_06.csv"]
            df = pd.concat([pd.read_csv(f"./original/chicago-bike/{_csv}") for _csv in csv])
            df['starttime'] = pd.to_datetime(df["starttime"], format='%m/%d/%Y %H:%M')
            df['stoptime'] = pd.to_datetime(df["stoptime"], format='%m/%d/%Y %H:%M')
            df = df[df['starttime'].apply(lambda r: r.year == 2016 and r.month in [4, 5, 6])]
            df = df[df['stoptime'].apply(lambda r: r.year == 2016 and r.month in [4, 5, 6])]
            pick = df[['starttime', 'from_station_id']].copy()
            drop = df[['stoptime', 'to_station_id']].copy()
            pick.columns = ["time", "station"]
            drop.columns = ["time", "station"]
            pick.dropna(inplace=True)
            drop.dropna(inplace=True)
            print(len(pick), len(drop))
            if mini_batch:
                pick = pick.head(100)
                drop = drop.head(100)

            df = pd.read_csv(f"./original/chicago-bike/Divvy_Stations_2016_Q1Q2.csv")
            df = df[["id", "latitude", "longitude"]]
            bike_stations_gps = {row.id: (row.latitude, row.longitude) for row in df.itertuples()}

            for i, data in enumerate([pick, drop]):
                def handle_row(row):
                    time = int((pd.to_datetime(row.time) - init_time) / delta)
                    station_gps = bike_stations_gps[row.station]
                    region = gps2region(station_gps[0], station_gps[1])
                    if region > 0:
                        flows['bike'][time, region, i] += 1


                tqdm.pandas(ncols=150, desc=f"Bike " + ["Pick", "Drop"][i])
                data.progress_apply(lambda row: handle_row(row), axis=1)

        elif dataset == 'taxi':
            # since original data is too huge, extract records only in 2016Q2
            if not os.path.exists(f"./original/chicago-taxi/Taxi_Trips_2016Q2.csv"):
                cs = 10000
                df_iter = pd.read_csv(f"./original/chicago-taxi/Taxi_Trips_-_2016.csv", chunksize=cs)

                data = pd.DataFrame()
                for _, df in enumerate(df_iter):
                    df = df[df.apply(lambda row: int(getattr(row, 'Trip Start Timestamp')[:2]) in [4, 5, 6], axis=1)]
                    print(f"Read {_ * cs} lines, get {len(data)}, add {len(df)} records")
                    data = pd.concat([data, df])
                print(f"Saving to ./original/chicago-taxi/Taxi_Trips_2016Q2.csv")
                data.to_csv(f"./original/chicago-taxi/Taxi_Trips_2016Q2.csv")

            df = pd.read_csv(f"./original/chicago-taxi/Taxi_Trips_2016Q2.csv")
            df['Trip Start Timestamp'] = pd.to_datetime(df["Trip Start Timestamp"])
            df['Trip End Timestamp'] = pd.to_datetime(df["Trip End Timestamp"])
            df = df[df['Trip Start Timestamp'].apply(lambda r: r.year == 2016 and r.month in [4, 5, 6])]
            df = df[df['Trip End Timestamp'].apply(lambda r: r.year == 2016 and r.month in [4, 5, 6])]
            pick = df[['Trip Start Timestamp', "Pickup Census Tract", "Pickup Centroid Latitude",
                       "Pickup Centroid Longitude"]].copy()
            drop = df[['Trip End Timestamp', "Dropoff Census Tract", "Dropoff Centroid Latitude",
                       "Dropoff Centroid Longitude"]].copy()
            pick.columns = ["time", "station", "latitude", "longitude"]
            drop.columns = ["time", "station", "latitude", "longitude"]
            pick.dropna(inplace=True)
            drop.dropna(inplace=True)
            print(len(pick), len(drop))
            if mini_batch:
                pick = pick.head(100)
                drop = drop.head(100)

            for i, data in enumerate([pick, drop]):
                def handle_row(row):
                    time = int((pd.to_datetime(row.time) - init_time) / delta)
                    station_gps = (row.latitude, row.longitude)
                    region = gps2region(station_gps[0], station_gps[1])
                    if region > 0:
                        flows['taxi'][time, region, i] += 1


                tqdm.pandas(ncols=150, desc=f"Taxi " + ["Pick", "Drop"][i])
                data.progress_apply(lambda row: handle_row(row), axis=1)

        else:
            raise ValueError()

    # flows = np.concatenate([flows['bike'], flows['taxi']], axis=-1)  # (n_time, n_region, 4)
    bike_idx = flows['bike'].sum(axis=0).sum(axis=1) != 0  # remove empty station
    taxi_idx = flows['taxi'].sum(axis=0).sum(axis=1) != 0
    idx = np.logical_and(bike_idx, taxi_idx)
    idx = np.where(idx)[0]

    print(len(idx))
    print(idx)
    flows_bike = flows['bike'][:, idx, :]
    flows_taxi = flows['taxi'][:, idx, :]
    flows_mix = np.concatenate([flows_bike, flows_taxi], -1)
    regions = [regions[i] for i in idx]

    print(f"Generate adjacency matrix")
    position_dict = [(region.centroid.y, region.centroid.x) for region in regions]
    adj_mx = get_adjacency_matrix(position_dict, normalized_k)

    # to df and set index
    time = np.array(pd.date_range(init_time, end_time, freq=freq).strftime('%Y-%m-%d %H:%M:%S'))
    f = h5py.File(f"./h5data/chicago-mix.h5", 'w')
    f.create_dataset("raw_data", data=flows_mix)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    f = h5py.File(f"./h5data/chicago-bike.h5", 'w')
    f.create_dataset("raw_data", data=flows_bike)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    f = h5py.File(f"./h5data/chicago-taxi.h5", 'w')
    f.create_dataset("raw_data", data=flows_taxi)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    print("end")
