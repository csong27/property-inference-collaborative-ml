import numpy as np
from sklearn.model_selection import train_test_split


def split_data_mt(X, y, n_workers):
    size = len(X) // n_workers

    splitted_X = []
    splitted_y = []
    for i in range(n_workers):
        xx = X[size * i: size * (i + 1)]
        yy = y[size * i: size * (i + 1), :]
        splitted_X.append(xx)
        splitted_y.append(yy)

    return splitted_X, splitted_y


def add_nonprop(test_prop_indices, nonprop_indices, p_prop=0.7):
    n = len(test_prop_indices)
    n_to_add = int(n / p_prop) - n
    print 'Adding {} non prop data to victim'.format(n_to_add)
    sampled_non_prop = np.random.choice(nonprop_indices, n_to_add, replace=False)
    nonprop_indices = np.setdiff1d(nonprop_indices, sampled_non_prop)
    return sampled_non_prop, nonprop_indices


def prepare_data_biased(data, train_size=0.5, n_workers=5, p_prop=0.5, shuffle=True, balance=True, seed=None,
                        victim_all_nonprop=False, test_size=0.3):
    if seed is not None:
        np.random.seed(seed)

    # only victim has biased data
    X, y, p = data

    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        p = p[indices]

    y_cat = np.asarray(list(zip(y, p)))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, random_state=seed, test_size=test_size, stratify=y)
    print "Training data size {}, testing data size {}".format(len(X_train), len(X_test))

    p_train = y_train[:, 1]
    n_train = len(p_train)

    indices = np.arange(n_train)
    prop_indices = indices[p_train == 1]
    nonprop_indices = indices[p_train == 0]

    len_p = len(prop_indices)
    prop_train_size = int(len_p * train_size)

    train_prop_indices = np.random.choice(prop_indices, prop_train_size, replace=False)
    victim_prop_indices = np.setdiff1d(prop_indices, train_prop_indices)

    if balance:
        n_per_worker = len(X_train) // n_workers
        p_prop_bal = float(len(victim_prop_indices)) / n_per_worker
        print len(victim_prop_indices), n_per_worker, p_prop_bal
        if p_prop_bal <= p_prop:
            p_prop = p_prop_bal
        print 'Balance split...victim has {:.4f} data with property'.format(p_prop)

    victim_nonprop_indices, nonprop_indices = add_nonprop(victim_prop_indices, nonprop_indices, p_prop)

    print 'Victim has {} prop {} nonprop'.format(len(victim_prop_indices), len(victim_nonprop_indices))

    # other participants only has non prop data
    splitted_X, splitted_y = split_data_mt(X_train[nonprop_indices], y_train[nonprop_indices], n_workers - 1)

    if victim_all_nonprop:
        victim_n_prop = len(victim_prop_indices)
        attacker_indices = np.arange(len(splitted_X[0]))
        to_replace_indices = np.random.choice(attacker_indices, victim_n_prop, replace=False)
        other_indices = np.setdiff1d(attacker_indices, to_replace_indices)
        victim_X = splitted_X[0][to_replace_indices]
        victim_y = splitted_y[0][to_replace_indices]
        # attacker can have some prop data
        splitted_X[0] = np.vstack([splitted_X[0][other_indices], X_train[victim_prop_indices],
                                   X_train[train_prop_indices]])
        splitted_y[0] = np.concatenate([splitted_y[0][other_indices], y_train[victim_prop_indices],
                                        y_train[train_prop_indices]])
        # victim is trainer 0
        splitted_X = [(victim_X, X_train[victim_nonprop_indices])] + splitted_X
        splitted_y = [(victim_y, y_train[victim_nonprop_indices])] + splitted_y
    else:
        # attacker can have some prop data
        splitted_X[0] = np.vstack([splitted_X[0], X_train[train_prop_indices]])
        splitted_y[0] = np.concatenate([splitted_y[0], y_train[train_prop_indices]])
        # victim is trainer 0
        splitted_X = [(X_train[victim_prop_indices], X_train[victim_nonprop_indices])] + splitted_X
        splitted_y = [(y_train[victim_prop_indices], y_train[victim_nonprop_indices])] + splitted_y

    return splitted_X, splitted_y, X_test, y_test
