#!/usr/bin/python
# -*- coding: UTF-8 -*-

from particleswarm.swarm import Swarm
import matplotlib.pyplot as plt
import numpy as np
import math

def get_sin_amplitude_plot(flag = True):
  surge_point = 1000
  amplitudeAfterSurge = 1000
  T = 140
  axes_x = np.linspace(0, surge_point, 1000)
  axes_y = []
  
  if flag :
    axes_y = np.concatenate([np.random.normal(0,1,210) * amplitudeAfterSurge, [(amplitudeAfterSurge) * np.abs(math.sin(2*math.pi*i/T)) for i in axes_x[210:800]],  np.random.normal(0,1,200) * amplitudeAfterSurge])
    # print(axes_y)
  else:
    axes_y = [(amplitudeAfterSurge - i) * math.sin(2*math.pi*i/T) for i in axes_x]
  
  plt.plot(axes_x, axes_y)
  plt.show()
  return axes_y

def generate_s(series, i , b, l):
  start = b + l*(i - 1)
  end = b + l*i - 1 if (b + l*i - 1) <= len(series) else len(series) - 1
  return series[start: end]

def E_s(data, b, e, l):
#   print(b)
#   print(e)
  if(e < 0 or b < 0):
      return 1000000
  if(e < b):
    e, b = b, e
  n = int(np.abs(e - b) // l)
  if n == 0:
    return 10000000
  error = 0
  for i in range(1, n):
    s_i = np.array(generate_s(data, i, int(b), int(l)))
    # print(s_i)
    # sleep(10)
    for j in range(i + 1, n):
      s_j = np.array(generate_s(data, j, int(b), int(l)))
    #   print(np.sum((s_i - s_j) ** 2))
      min_len = min(len(s_i), len(s_j))
      error += np.sum((s_i[:min_len] - s_j[:min_len]) ** 2)
  
  error = float(error)/(n * l)
  return error


def E_t(x, data, l):
  b, e = x
  data = data
  l = l
  # l = 70
  return 0.1 * E_s(data, b, e, l) + 100/((e - b) + 1)


class My_Sworm(Swarm):
    def __init__ (self, 
            swarmsize, 
            minvalues, 
            maxvalues, 
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio):
        self.data = get_sin_amplitude_plot(True)
        
        Swarm.__init__ (self, 
            swarmsize, 
            minvalues, 
            maxvalues, 
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio)


    def _finalFunc (self, position):
        # function = sum (-position * np.sin (np.sqrt (np.abs (position) ) ) )
        args = (self.data, 70)
        function = E_t(position, self.data, 70) + E_t(len(self.data) - position, self.data[::-1], 70)
        # mistake = self._getPenalty(position, 10000.0)
        # print("Penalty", penalty)
        # return function + mistake
        return function