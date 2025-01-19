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
    
    truth_table = x[5:69][:]
    xor_table = numpy.zeros((6, truth_table.shape[1]))
    for b in range(0, 6):
        for i in range(0, 32):
            tail = i & ((1 << b) - 1)
            head = i >> b
            index0 = (head << (b + 1)) + (0 << b) + tail
            index1 = (head << (b + 1)) + (0 << b) + tail
            xor_table[b][:] += numpy.logical_xor(truth_table[index0], truth_table[index1])
    
    x = numpy.vstack([x, xor_table])
    
    return x
