#!/usr/bin/python
# -*- coding: UTF-8 -*-

from time_series_task import My_Sworm
from utils import printResult
import matplotlib.pyplot as plt

import numpy


if __name__ == "__main__":
    iterCount = 10

    dimension = 2
    swarmsize = 10

    minvalues = numpy.array ([0] * dimension)
    maxvalues = numpy.array ([1000] * dimension)

    currentVelocityRatio = 0.2
    localVelocityRatio = 2.8
    globalVelocityRatio = 1.3

    swarm = Swarm_Schwefel (swarmsize, 
            minvalues, 
            maxvalues,
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio
            )
    # get_sin_amplitude_plot()
    for n in range (iterCount):
        for i in range(swarmsize):
            print(f"Position_{i} = ", swarm[i].position)
            print(f"Velocity_{i} = ", swarm[i].velocity)

        print(printResult(swarm, n))

        swarm.nextIteration()
