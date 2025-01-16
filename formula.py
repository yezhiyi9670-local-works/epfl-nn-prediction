import numpy

def f1_score(pred, gt):
    gt0_pred1 = numpy.sum(numpy.logical_and(pred == 1, gt == 0))
    gt1_pred0 = numpy.sum(numpy.logical_and(pred == 0, gt == 1))
    gt1_pred1 = numpy.sum(numpy.logical_and(pred == 1, gt == 1))
    
    if gt1_pred1 == 0:
        return 0
    
    if gt1_pred1 + gt0_pred1 == 0:
        return numpy.nan
    if gt1_pred1 + gt1_pred0 == 0:
        return numpy.nan
    
    precision = gt1_pred1 / (gt1_pred1 + gt0_pred1)
    recall = gt1_pred1 / (gt1_pred1 + gt1_pred0)
    if precision == 0 or recall == 0:
        return 0
    
    return 1 / ((1 / precision + 1 / recall) / 2)
    