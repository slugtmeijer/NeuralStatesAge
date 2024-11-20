import numpy as np


def compute_overlap_1swindow(mode: int, data: np.ndarray, events: np.ndarray,events1: np.ndarray,eventsm1: np.ndarray):
    # This function computes the boundary overlap between between regions and events
    # mode 1 = relative overlap between regions and events (only works correctly with binary boundaries!)
    # mode 2 = absolute overlap between regions and events (scaled wrt the number of boundaries in a region -
    # mode 2 works with any boundary value (i.e. boundary strength)

    # argument 2 is the data per region (region x time)
    # argument 3 is the event timecourse (time x 1)
    # argument 4 is the event timecourse, created after shifting all onset times one second in the future (time x 1)
    # argument 5 is the event timecourse, created after shifting all onset times one second in the past (time x 1)

    Ntime = data.shape[1]
    Nregs = data.shape[0]
    sumbounds = np.expand_dims(np.sum(data, 1), axis=1)
    ns = np.transpose(sumbounds / Ntime)

    #sum all unique event boundaries (across all +1/0/-1 delays)
    allevents = (events+events1+eventsm1)>0
    #compute expected number of overlaps based on that total sum
    ns2 = np.ones((1, Nregs)) * np.sum(allevents) / Ntime
    expected_overlap = ns * ns2 * Ntime

    #find the TRs where event boundaries are. The order of events boundaries is assumed to be the same across the three inputs
    events_loc_s0 = np.where(events>0)[1]
    events_loc_s1 = np.where(events1>0)[1]
    events_loc_sm1 = np.where(eventsm1 > 0)[1]

    #isolate the timepoints that take place at event boundaries and stack them in a 3D matrix
    data_at_eventboundaries = np.stack((data[:,events_loc_s0],data[:, events_loc_s1],data[:, events_loc_sm1]), axis=2)

    #for each event boundary, look at its maximum overlap with neural states across the three shifts
    #Next, sum the overlap over all events
    real_overlap = np.sum(np.max(data_at_eventboundaries,axis=2), axis=1)

    # tile the events timeseries, so that they are Nregs * Ntime
    #events_tileds0 = np.tile(events, [Nregs, 1])
    #events_tileds1 = np.tile(events1, [Nregs, 1])
    #events_tiledsm1 = np.tile(eventsm1, [Nregs, 1])

    #Multiply each events time series with the neural state timeseries
    #real_overlap_s0 = np.multiply(events_tileds0, data)
    #real_overlap_s1 = np.multiply(events_tileds1, data)
    #real_overlap_sm1 = np.multiply(events_tiledsm1, data)
    #real_overlap = np.sum((real_overlap_sm1 + real_overlap_s1 + real_overlap_s0) > 0, axis=1)

    # real_overlap = np.transpose(np.matmul(data, np.transpose(events)))

    if mode == 1:

        # take the smallest of 1. number of states per region and 2. number of event boundaries
        ev = np.expand_dims(np.tile(np.sum(events), (Nregs)), axis=1)
        maximum_overlap = np.min(np.concatenate((sumbounds, ev), axis=1), axis=1)
        overlap = (real_overlap - expected_overlap) / (maximum_overlap - expected_overlap)

    elif mode == 2:
        maximum_overlap = np.transpose(sumbounds)
        overlap = (real_overlap - expected_overlap) / (maximum_overlap - expected_overlap)

    return overlap