import pandas as pd
from tqdm import tqdm
import json

# f = open('../xfdata/sensor_graph/SensorIds.txt', 'r')
# geo_ids = f.read()
# geo_ids = geo_ids.strip().split(',')
# geo_ids = list(map(eval, geo_ids))
#
# geo = []
# for i in range(len(geo_ids)):
#     geo.append([geo_ids[i], 'Point', '[]'])
# geo = pd.DataFrame(geo, columns=['geo_id', 'type', 'coordinates'])
# geo.to_csv('../user_data/tmp_data/xunfei.geo', index=False)
#
# links = pd.read_csv('../xfdata/sensor_graph/SensorDistances.csv')
# rel = []
# cnt = 0
# for i in tqdm(range(len(links))):
#     fr, to, cost = int(links.iloc[i]['from']), int(links.iloc[i]['to']), float(links.iloc[i]['cost'])
#     if fr not in geo_ids or to not in geo_ids:
#         continue
#     rel.append([cnt, 'geo', fr, to, cost])
#     cnt += 1
# rel = pd.DataFrame(rel, columns=['rel_id', 'type', 'origin_id', 'destination_id', 'cost'])
# rel.to_csv('../user_data/tmp_data/xunfei.rel', index=False)

config = {
  "info": {
    "data_col": [
      "traffic_speed"
    ],
    "weight_col": "cost",
    "data_files": [
      "xunfei"
    ],
    "geo_file": "xunfei",
    "rel_file": "xunfei",
    "output_dim": 1,
    "time_intervals": 300,
    "init_weight_inf_or_zero": "inf",
    "set_weight_link_or_dist": "dist",
    "calculate_weight_adj": True,
    "weight_adj_epsilon": 0.1
  }
}
json.dump(config, open('../user_data/tmp_data/config.json', 'w'), indent=4)
