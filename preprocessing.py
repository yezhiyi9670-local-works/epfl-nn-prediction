import numpy

def preprocess_input(x: numpy.ndarray):
    x[0:2][:] = 0
    x[4:5][:] = 0
    # levels = x[2]
    # levelsR = x[3]
    # x[2][:] = (levels) / (levels + levelsR + 1e-5)
    # x[3][:] = (levelsR) / (levels + levelsR + 1e-5)
    x[2:4][:] = 0
    return x
