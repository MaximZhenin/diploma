from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat
from src.autoperiod.autoperiod import Autoperiod
from src.autoperiod.helpers import *
from src.autoperiod.plotting import Plotter
from collections import Counter
import math as m
import logging
from matplotlib import pyplot
from clustering.dbscan import dbscan
from checkmonotonic import checkOut
import warnings
import csv
import os.path


warnings.filterwarnings("ignore")


features_288_temp_1 = {"l1_cache_miss" : 0,
              "llc_cache_miss": 1,
              "mem_load" : 2,
              "cpu_user" :3,
              "mem_store":4,
              "ib_rcv_data_mpi":5,
              "ib_rcv_pckts_mpi":6,
              "ib_xmit_data_mpi":7,
              "ib_xmit_pckts_mpi":8,
              "loadavg":9,
              "cpu_nice":10,
              "ib_rcv_data_fs":11,
              "ib_rcv_pckts_fs":12,
              "ib_xmit_data_fs":13,
              "ib_xmit_pckts_fs":14,
              "gpu_load":15
            }


def get_rid_off_clones(arr):
  several_ans = np.array(list(x[1] for x in arr))
  clusterRes_avg, count = dbscan(several_ans.reshape(-1,1), 0.1, 1)
  # print(clusterRes_avg)
  
  two_periods = np.unique(clusterRes_avg)
  period = np.asarray([])
  for i in two_periods[:2]:
    # print(np.mean(several_ans[clusterRes_avg==i]))
    period = np.append(period, np.mean(several_ans[clusterRes_avg==i]) )

  return period


def writeCSV(func):
    def wrapcsv(*args,**kwargs):
        file_exists = os.path.exists('MonitoringResult.csv')
        with open("MonitoringResult.csv", "a", newline='') as outfile:
            writer = csv.writer(outfile)
            if not file_exists:
              writer.writerow(['JobId', 'Period1', 'Period2', 'IsMonoton'])
            funcval= func(*args,**kwargs)
            print(funcval)
            writer.writerow(funcval)
        return funcval
    return wrapcsv

@writeCSV
def task(jobid, job, lst_features, all_features=features_288_temp_1, ma='AVG'):
  first3 = getTwoPeriod(jobid, job, lst_features, all_features, ma)
  print(first3)
  if first3[1] == 0.0:
    last = False
  else:
    last = checkOut(job, int(first3[1]), lst_features, all_features, ma)
  return [*first3, last]

def getTwoPeriod(jobid, job, lst_features, all_features=features_288_temp_1, ma='AVG'):
  step = 2
  if ma == 'AVG':
    start = 0
  else:
    start = 1
  
  y_pred_1 = np.asarray([])
  y_pred_2 = np.asarray([])
  for i in range(start, job.shape[0], step):
    times, values = np.arange(job.shape[1]), job[i]
    p = Autoperiod(times, values, plotter=None)
    periods = get_rid_off_clones(p.get_filter_valid_periods)
    if len(periods) == 0:
      y_pred_1 = np.append(y_pred_1, 0)
      y_pred_2 = np.append(y_pred_2, 0)
    elif len(periods) == 1:
      y_pred_1 = np.append(y_pred_1, periods[0])
      y_pred_2 = np.append(y_pred_2, 0)
    else:
      y_pred_1 = np.append(y_pred_1, periods[0])
      y_pred_2 = np.append(y_pred_2, periods[1])
  ans = np.asarray([])
  for y_pred in (y_pred_1, y_pred_2):
    temp_y_pred = np.asarray([])
    for ftr in lst_features:
      index = all_features.get(ftr)
      if index != None:
        try:
          value = y_pred[index]
          temp_y_pred = np.append(temp_y_pred, value)
        except Exception as e:
          logging.exception("Wrong Input Data")
      else:
        print("no such key = " + ftr)
    radius = 0.1
    min_num_of_cluster = 2
    clusterRes_avg, count = dbscan(temp_y_pred.reshape(-1,1), radius, min_num_of_cluster)
    max_label = Counter(clusterRes_avg).most_common()[0]
    max_label = max_label[0]
    if count == len(lst_features) or max_label == -1:
      ans = np.append(ans, 0)
    else:
      ans = np.append(ans, np.mean(temp_y_pred[clusterRes_avg == max_label]))
  return [jobid, ans[0], ans[1]]


if __name__ == '__main__':
  print("Start")
  b = np.load('job_data_original/job_data_original/1353891.npy', 'r')
  temp = b
  ans_new = task('1353891', temp, ['l1_cache_miss', 'llc_cache_miss', 'cpu_user', 'loadavg', 'ib_rcv_pckts_fs', 'ib_xmit_pckts_fs'])
  print("two_period_alg_ans", ans_new)