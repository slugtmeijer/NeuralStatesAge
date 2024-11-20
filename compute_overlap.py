import numpy as np

def compute_overlap(mode:int, data:np.ndarray, events:np.ndarray):
    #This function computes the boundary overlap between between regions and events
    #mode 1 = relative overlap between regions and events (only works correctly with binary boundaries!)
    #mode 2 = absolute overlap between regions and events (scaled wrt the number of boundaries in a region -
    #mode 2 works with any boundary value (i.e. boundary strength)
    
    #argument 2 is the data per region (region x time)
    #argument 3 is the event timecourse (time x 1)

    Ntime = data.shape[1]
    Nregs = data.shape[0]
    sumbounds = np.expand_dims(np.sum(data,1), axis=1)
    ns = np.transpose(sumbounds/Ntime)
    ns2 = np.ones((1, Nregs)) * np.sum(events) / Ntime
    expected_overlap = ns * ns2 * Ntime
    real_overlap = np.transpose(np.matmul(data, np.transpose(events)))

    if mode == 1:

        #take the smallest of 1. number of states per region and 2. number of event boundaries
        ev = np.expand_dims(np.tile(np.sum(events),(Nregs)), axis=1)
        maximum_overlap = np.min(np.concatenate((sumbounds,ev), axis=1), axis=1)
        overlap = (real_overlap-expected_overlap)/(maximum_overlap-expected_overlap)

    elif mode == 2:
        maximum_overlap = np.transpose(sumbounds)
        overlap = (real_overlap-expected_overlap)/(maximum_overlap-expected_overlap)

    return overlap