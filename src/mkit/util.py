import numpy
def str2bool(string):
    if string == 'True':
        return True
    return False

def index_max(arr):
    return numpy.argmax(arr)

def index_min(arr):
    return arr.index(min(arr))