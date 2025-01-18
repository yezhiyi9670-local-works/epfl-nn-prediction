import numpy

def preprocess_input(x: numpy.ndarray):
    fanin = x[0][:]
    fanout = x[1][:]
    x[0][:] = numpy.arctan2(fanout, fanin)
    x[1][:] = numpy.arctan2(fanin, fanout)
    # x[0:2][:] = 0
    
    levels = x[2]
    levelsR = x[3]
    x[2][:] = (levels) / (levels + levelsR + 1e-5)
    x[3][:] = (levelsR) / (levels + levelsR + 1e-5)
    # x[2:4][:] = 0
    
    x[4:5][:] = 0
    
    return x
