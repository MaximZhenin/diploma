#!/usr/bin/python
# -*- coding: UTF-8 -*-

from swarm_rastrigin import Swarm_Rastrigin
from utils import printResult

import numpy


if __name__ == "__main__":
    iterCount = 100

    dimension = 2
    swarmsize = 2000

    minvalues = numpy.array ([-5.12] * dimension)
    maxvalues = numpy.array ([5.12] * dimension)

    currentVelocityRatio = 0.5
    localVelocityRatio = 2.0
    globalVelocityRatio = 5.0

    swarm = Swarm_Rastrigin (swarmsize, 
            minvalues, 
            maxvalues,
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio
            )

    for n in range (iterCount):
        print("Position", swarm[0].position)
        print("Velocity", swarm[0].velocity)

        print(printResult(swarm, n))

        swarm.nextIteration()
