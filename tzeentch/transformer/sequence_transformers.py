import random
from collections import deque
import numpy as np
import pywt

def sequence_generator(df, TIME_SEQ_LEN, shuffle=True, seed=101):
    sequential_data = []  # this is a list that will CONTAIN the sequences
    queue = deque(
            maxlen=TIME_SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        queue.append([n for n in i[:-1]])  # store all but the target
        if len(queue) == TIME_SEQ_LEN:  # make sure we have 60 sequences!

            # wavelets noise share
            x = np.array(queue)
            (ca, cd) = pywt.dwt(x, "haar")
            cat = pywt.threshold(ca, np.std(ca), mode="soft")
            cdt = pywt.threshold(cd, np.std(cd), mode="soft")
            tx = pywt.idwt(cat, cdt, "haar")

            sequential_data.append([tx, i[-1]])  # append those bad boys!

    if shuffle:
        random.seed(seed)
        random.shuffle(sequential_data)  # shuffle for good measure.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!