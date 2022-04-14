#!/usr/bin/python
# -*- coding: UTF-8 -*-

from particleswarm.swarm import Swarm


class Swarm_X2 (Swarm):
    def __init__ (self, 
            swarmsize, 
            minvalues, 
            maxvalues, 
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio):
       Swarm.__init__ (self, 
            swarmsize, 
            minvalues, 
            maxvalues, 
            currentVelocityRatio,
            localVelocityRatio, 
            globalVelocityRatio)


    def _finalFunc (self, position):
        penalty = self._getPenalty (position, 10000.0)
        finalfunc = sum (position * position)

        return finalfunc + penalty
