import os
import math
import h5py
import shapefile
import numpy as np
import pandas as pd
import multiprocessing
from pyproj import CRS, Transformer
import shapely.geometry as geometry
from gen_h5_nyc import get_adjacency_matrix

date_range = ["7/1/2017 00:00:00", "9/30/2017 23:59:59"]
delta = np.timedelta64(30, "m")
freq = "30min"
mini_batch = False
n_process = int(os.cpu_count() * 0.9)


def read_geo(proj_file_path, shape_file_path):
    with open(proj_file_path) as prj_f:
        prj = prj_f.read()
    world_crs = CRS.from_epsg(4326)
    beijing_crs = CRS.from_string(prj)
    world2beijing = Transformer.from_crs(world_crs, beijing_crs)
    beijing2world = Transformer.from_crs(beijing_crs, world_crs)
    sf = shapefile.Reader(shape_file_path)
    regions = [geometry.shape(s) for s in sf.shapes()]

    def gps2region(lat, lon):
        p = world2beijing.transform(lat, lon)  # world to beijing
        p = geometry.Point(p)
        for i, region in enumerate(regions):
            if region.contains(p):
                return i
        return -1

    return gps2region, regions, beijing2world


def map_filter_csvs(p_id, dir_path, csv_list):
    csv_id = 0
    data = pd.DataFrame()
    for _, f in enumerate(csv_list):
        df = pd.read_csv(os.path.join(dir_path, f),
                         usecols=[2, 4, 6, 7, 9, 11, 12], encoding_errors='ignore', on_bad_lines='skip')
        df.columns = ["mode", "starttime", "start_lon", "start_lat", "stoptime", "stop_lon", "stop_lat"]
        print(f"p{p_id}\t[{_ + 1}/{len(csv_list)}]\t{f}, {len(df)} records total")

        df['starttime'] = pd.to_datetime(df["starttime"], errors='coerce')
        df['stoptime'] = pd.to_datetime(df["stoptime"], errors='coerce')
        df['start_lon'] = pd.to_numeric(df['start_lon'], downcast='float', errors='coerce')
        df['start_lat'] = pd.to_numeric(df['start_lat'], downcast='float', errors='coerce')
        df['stop_lon'] = pd.to_numeric(df['stop_lon'], downcast='float', errors='coerce')
        df['stop_lat'] = pd.to_numeric(df['stop_lat'], downcast='float', errors='coerce')
        df.dropna(inplace=True)
        df = df[df['mode'].isin(['R', 'B'])]
        df = df[(df['starttime'].dt.year == 2017) & (df['starttime'].dt.month.isin([7, 8, 9]))]
        df = df[(df['stoptime'].dt.year == 2017) & (df['stoptime'].dt.month.isin([7, 8, 9]))]
        data = pd.concat([data, df])

        if len(data) > 1e6:  # avoid memory overflow
            path = os.path.join(dir_path, "2017Q3", f"p{p_id}-{csv_id}.csv")
            print(f"p{p_id}\tsave {len(data)} records to {path}")
            data.to_csv(path, index=False)
            data = pd.DataFrame()
            csv_id += 1

    if len(data) > 0:
        path = os.path.join(dir_path, "2017Q3", f"p{p_id}-{csv_id}.csv")
        print(f"p{p_id}\tsave {len(data)} to {path}")
        data.to_csv(path, index=False)
    print(f"p{p_id}\tfinished")


def map_aggregate(p_id, num_time, num_regions, dir_path, csv_list):
    flows = {'railway': np.zeros((num_time, num_regions, 2), dtype=int),
             'bus': np.zeros((num_time, num_regions, 2), dtype=int)}
    mode_dict = {"R": "railway", "B": "bus"}
    count = {"n_records": 0, "n_drop_records": 0}

    for _, csv in enumerate(csv_list):
        try:
            df = pd.read_csv(os.path.join(dir_path, csv))
            df = df.head(10000) if mini_batch else df
            print(f"p{p_id}\t[{_ + 1}/{len(csv_list)}]\t{csv}, {len(df)} records total")
            df['starttime'] = pd.to_datetime(df["starttime"], errors='coerce')
            df['stoptime'] = pd.to_datetime(df["stoptime"], errors='coerce')
            df.dropna(inplace=True)
            pick = df[['mode', 'starttime', 'start_lat', 'start_lon']].copy()
            drop = df[['mode', 'stoptime', 'stop_lat', 'stop_lon']].copy()
            pick.columns = ['mode', "time", "latitude", "longitude"]
            drop.columns = ['mode', "time", "latitude", "longitude"]
            print(f"p{p_id}\tpick:{len(pick)}\tdrop:{len(drop)}")

            for i, data in enumerate([pick, drop]):  # most consuming
                def handle_row(row):
                    time = int((pd.to_datetime(row['time']) - init_time) / delta)
                    station_gps = (row['latitude'], row['longitude'])
                    region = gps2region(station_gps[0], station_gps[1])
                    if region > 0:
                        flows[mode_dict[row['mode']]][time, region, i] += 1
                        count["n_records"] += 1
                    else:
                        count["n_drop_records"] += 1

                data.apply(lambda row: handle_row(row), axis=1)
        except Exception as e:
            os.makedirs(f"./original/Beijing_IC_record/error/", exist_ok=True)
            with open(f"./original/Beijing_IC_record/error/{p_id}-{csv}.log", 'w') as io:
                io.write(f"{str(e)} from {p_id} in {csv}")

    return flows, count


def reduce_aggregate(flows_list, num_time, num_regions):
    flows = {'railway': np.zeros((num_time, num_regions, 2), dtype=int),
             'bus': np.zeros((num_time, num_regions, 2), dtype=int)}
    n_records, n_drop_records = 0, 0

    for flow, c in flows_list:
        flows['railway'] += flow['railway']
        flows['bus'] += flow['bus']
        n_records += c['n_records']
        n_drop_records += c['n_drop_records']
    return flows, (n_records, n_drop_records)


if __name__ == '__main__':
    init_time = pd.to_datetime(date_range[0], format='%m/%d/%Y %H:%M:%S')
    end_time = pd.to_datetime(date_range[1], format='%m/%d/%Y %H:%M:%S')
    num_time = math.ceil((end_time - init_time) / delta)

    gps2region, regions, beijing2world = read_geo(
        proj_file_path='./original/Beijing_IC_record/tracts/北京乡镇边界.prj',
        shape_file_path='./original/Beijing_IC_record/tracts/北京乡镇边界.shp',
    )

    dir_path = './original/Beijing_IC_record/'
    csv_path = os.path.join(dir_path, "2017Q3")
    os.makedirs(csv_path, exist_ok=True)
    # since original data is too huge, extract records only in 2017Q3
    if len(os.listdir(csv_path)) <= 0:
        data = pd.DataFrame()
        csv = os.listdir(dir_path)
        csv = list(
            filter(lambda c: os.path.getsize(os.path.join(dir_path, c)) > 0 and (c[0] == 't' and 'csv' in c), csv))
        bs = math.ceil(len(csv) / n_process)

        jobs = []
        pool = multiprocessing.Pool(processes=n_process)
        for p_id in range(n_process):
            csvs = csv[p_id * bs:(p_id + 1) * bs]
            jobs.append(pool.apply_async(map_filter_csvs, (p_id, dir_path, csvs)))
        pool.close()
        pool.join()
        print("Generate 2017Q3/ done")
        exit(0)

    # read csv and aggregate traffic flow
    print(f"Read files from {csv_path}")
    csvs = os.listdir(csv_path)
    bs = math.ceil(len(csvs) / n_process)
    pool = multiprocessing.Pool(processes=n_process)
    jobs = []
    for p_id in range(n_process):
        csv_list = csvs[p_id * bs:(p_id + 1) * bs]
        jobs.append(pool.apply_async(map_aggregate, (p_id, num_time, len(regions), csv_path, csv_list,)))
    pool.close()
    pool.join()
    print("Aggregate traffic flow done")

    flows_list = [j.get() for j in jobs]
    flows, (n_records, n_drop_records) = reduce_aggregate(flows_list, num_time, len(regions))
    print(f"n_records:{n_records}, n_drop_records:{n_drop_records}")

    # drop empty regions
    railway_idx = flows['railway'].sum(axis=0).sum(axis=1) != 0
    bus_idx = flows['bus'].sum(axis=0).sum(axis=1) != 0
    idx = np.logical_and(railway_idx, bus_idx)
    idx = np.where(idx)[0]

    print(len(idx))
    print(idx)
    flows_railway = flows['railway'][:, idx, :]
    flows_bus = flows['bus'][:, idx, :]
    flows_mix = np.concatenate([flows_railway, flows_bus], -1)
    regions = [regions[i] for i in idx]

    print(f"Generate adjacency matrix")
    position_dict = [beijing2world.transform(region.centroid.x, region.centroid.y) for region in regions]
    adj_mx = get_adjacency_matrix(position_dict, 0)

    # to df and set index
    time = np.array(pd.date_range(init_time, end_time, freq=freq).strftime('%Y-%m-%d %H:%M:%S'))
    f = h5py.File(f"./h5data/beijing-mix.h5", 'w')
    f.create_dataset("raw_data", data=flows_mix)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    f = h5py.File(f"./h5data/beijing-railway.h5", 'w')
    f.create_dataset("raw_data", data=flows_railway)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    f = h5py.File(f"./h5data/beijing-bus.h5", 'w')
    f.create_dataset("raw_data", data=flows_bus)
    f.create_dataset("time", data=time)
    f.create_dataset("adjacency_matrix", data=adj_mx)

    print("end")
