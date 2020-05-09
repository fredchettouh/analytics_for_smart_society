def segmentation(data,window_size = 16):

    data['index_col'] = data.index
    counter=0
    data['window_id'] = np.nan 

    for (start, end) in windows2(data["index_col"], window_size):

        if(len(tr["index_col"][start:end]) == window_size):
            data['window_id'][start:end]=counter
            counter=counter+1

    data.fillna(method='ffill', inplace=True)
    data.drop(columns='index_col', inplace=True)
            
    return data
def windows2(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)
