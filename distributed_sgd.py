import os
import sys
import time
import lasagne
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict

from split_data import prepare_data_biased
from load_lfw import load_lfw_with_attrs, BINARY_ATTRS, MULTI_ATTRS

SAVE_DIR = './grads/'


if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def build_cnn_feat_extractor(input_var=None, input_shape=(None, 3, 50, 50), n=128):
    assert isinstance(n, int)
    network = OrderedDict()
    network['input'] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network['conv1'] = lasagne.layers.Conv2DLayer(
        network['input'], num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1'], pool_size=(2, 2))

    network['conv2'] = lasagne.layers.Conv2DLayer(
        network['pool1'], num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)
    network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2'], pool_size=(2, 2))

    network['conv3'] = lasagne.layers.Conv2DLayer(
        network['pool2'], num_filters=n, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)
    network['pool3'] = lasagne.layers.MaxPool2DLayer(network['conv3'], pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network['fc1'] = lasagne.layers.DenseLayer(
        network['pool3'],
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    return network


def build_mt_cnn(input_var=None, classes=2, infer_classes=2, input_shape=(None, 3, 50, 50), n=128):
    network = build_cnn_feat_extractor(input_var, input_shape, n)
    network['fc2'] = lasagne.layers.DenseLayer(
        network['fc1'],
        num_units=classes,
        nonlinearity=lasagne.nonlinearities.linear)

    network['fc2_B'] = lasagne.layers.DenseLayer(
        network['fc1'],
        num_units=infer_classes,
        nonlinearity=lasagne.nonlinearities.linear)

    return network


def build_cnn(input_var=None, classes=2, input_shape=(None, 3, 50, 50), n=128):
    network = build_cnn_feat_extractor(input_var, input_shape, n)
    network['fc2'] = lasagne.layers.DenseLayer(
        network['fc1'],
        num_units=classes,
        nonlinearity=lasagne.nonlinearities.linear)
    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, targets_B=None):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets_B is None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], targets_B[excerpt]


def train_lfw(task='gender', attr="race", prop_id=2, p_prop=0.5, n_workers=2, num_iteration=3000,
              alpha_B=0., victim_all_nonprop=False, balance=False, k=5, train_size=0.3):

    x, y, prop = load_lfw_with_attrs(task, attr)
    prop_dict = MULTI_ATTRS[attr] if attr in MULTI_ATTRS else BINARY_ATTRS[attr]

    print 'Training {} and infering {} property {} with {} data'.format(task, attr, prop_dict[prop_id], len(x))

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    prop = np.asarray(prop, dtype=np.int32)

    indices = np.arange(len(x))
    prop_indices = indices[prop == prop_id]
    nonprop_indices = indices[prop != prop_id]

    prop[prop_indices] = 1
    prop[nonprop_indices] = 0

    filename = "lfw_psMT_{}_{}_{}_alpha{}_k{}".format(task, attr, prop_id, alpha_B, k)

    if n_workers > 2:
        filename += '_n{}'.format(n_workers)

    train_multi_task_ps((x, y, prop), input_shape=(None, 3, 62, 47), p_prop=p_prop, balance=balance,
                        filename=filename, n_workers=n_workers, alpha_B=alpha_B, k=k,
                        num_iteration=num_iteration, victim_all_nonprop=victim_all_nonprop,
                        train_size=train_size)


def build_worker_attacker(input_shape, classes=2, infer_classes=2, lr=None, seed=54321, alph_B=0.5):
    lasagne.random.set_rng(np.random.RandomState(seed))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    target_var_B = T.ivector('targetsB')

    network_dict = build_mt_cnn(input_var, classes=classes, infer_classes=infer_classes, input_shape=input_shape)

    network, network_B = network_dict['fc2'], network_dict["fc2_B"]
    prediction, prediction_B = lasagne.layers.get_output([network, network_B])

    prediction = lasagne.nonlinearities.softmax(prediction)
    prediction_B = lasagne.nonlinearities.softmax(prediction_B)

    loss_A = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss_B = lasagne.objectives.categorical_crossentropy(prediction_B, target_var_B)

    loss = (1 - alph_B) * loss_A.mean() + alph_B * loss_B.mean()

    # save init
    params = lasagne.layers.get_all_params([network, network_B], trainable=True)

    if lr is not None:
        updates = lasagne.updates.sgd(loss, params, lr)
        train_fn = theano.function([input_var, target_var, target_var_B], loss, updates=updates)
        return params, train_fn

    grads = T.grad(loss, params)
    params_B = params[-2:]
    params = params[:-2]
    grads_B = grads[-2:]
    grads = grads[:-2]

    p_idx = 0
    grads_dict = dict()
    for p, g in zip(params, grads):
        key = p.name + str(p_idx)
        p_idx += 1
        grads_dict[key] = g

    grad_fn = theano.function([input_var, target_var, target_var_B], grads_dict)
    grads_B_fn = theano.function([input_var, target_var, target_var_B], grads_B)

    test_acc = T.sum(T.eq(T.argmax(prediction_B, axis=1), target_var_B), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var_B], [loss_B, test_acc])

    return params, grad_fn, params_B, grads_B_fn, val_fn


def build_worker(input_shape, classes=2, lr=None, seed=54321):
    lasagne.random.set_rng(np.random.RandomState(seed))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network_dict = build_cnn(input_var, classes=classes, input_shape=input_shape)
    network = network_dict['fc2']
    prediction = lasagne.layers.get_output(network)
    prediction = lasagne.nonlinearities.softmax(prediction)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # save init
    params = lasagne.layers.get_all_params(network, trainable=True)

    if lr is not None:
        updates = lasagne.updates.sgd(loss, params, lr)
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        return params, train_fn

    grads = T.grad(loss, params)

    p_idx = 0
    grads_dict = dict()
    for p, g in zip(params, grads):
        key = p.name + str(p_idx)
        p_idx += 1
        grads_dict[key] = g

    grad_fn = theano.function([input_var, target_var], grads_dict)
    return params, grad_fn


def inf_data(x, y, batchsize, shuffle=False, y_b=None):
    while True:
        for b in iterate_minibatches(x, y, batchsize=batchsize, shuffle=shuffle, targets_B=y_b):
            yield b


def mix_inf_data(p_inputs, p_targets, np_inputs, np_targets, batchsize, mix_p=0.5):
    p_batchsize = int(mix_p * batchsize)
    np_batchsize = batchsize - p_batchsize

    print 'Mixing {} prop data with {} non prop data'.format(p_batchsize, np_batchsize)

    p_gen = inf_data(p_inputs, p_targets, p_batchsize, shuffle=True)
    np_gen = inf_data(np_inputs, np_targets, np_batchsize, shuffle=True)

    while True:
        px, py = p_gen.next()
        npx, npy = np_gen.next()
        x = np.vstack([px, npx])
        y = np.concatenate([py, npy])
        yield x, y


def set_local(global_params, local_params_list):
    for params in local_params_list:
        for p, gp in zip(params, global_params):
            p.set_value(gp.get_value())


def update_global(global_params, grads, lr):
    for p, g in zip(global_params, grads):
        p_val = p.get_value()
        g = np.asarray(g)
        p.set_value(p_val - g * np.float32(lr))


def add_nonprop(test_prop_indices, nonprop_indices, p_prop=0.7):
    n = len(test_prop_indices)
    n_to_add = int(n / p_prop) - n

    sampled_non_prop = np.random.choice(nonprop_indices, n_to_add, replace=False)
    nonprop_indices = np.setdiff1d(nonprop_indices, sampled_non_prop)
    return sampled_non_prop, nonprop_indices


def gradient_getter(data, p_g, p_indices, fn, batch_size=32, shuffle=True):
    X, y = data
    p_x, p_y = X[p_indices], y[p_indices]

    for batch in iterate_minibatches(p_x, p_y, batch_size, shuffle=shuffle):
        xx, yy = batch
        gs = fn(xx, yy)
        p_g.append(np.asarray(gs).flatten())


def gradient_getter_with_gen(data_gen, p_g, fn, iters=10, param_names=None):
    for _ in range(iters):
        xx, yy = next(data_gen)
        gs = fn(xx, yy)
        if isinstance(gs, dict):
            gs = collect_grads(gs, param_names)
        else:
            gs = np.asarray(gs).flatten()
        p_g.append(gs)


def gradient_getter_with_gen_multi(data_gen1, data_gen2, p_g, fn, iters=10, n_workers=5, param_names=None):
    for _ in range(iters):
        xx, yy = next(data_gen1)
        pgs = fn(xx, yy)

        if isinstance(pgs, dict):
            for key in pgs:
                pgs[key] = np.asarray(pgs[key])
        else:
            pgs = np.asarray(pgs).flatten()

        for _ in range(n_workers - 2):
            xx, yy = next(data_gen2)
            npgs = fn(xx, yy)
            if isinstance(npgs, dict):
                for key in npgs:
                    pgs[key] += np.asarray(npgs[key])
            else:
                npgs = np.asarray(npgs).flatten()
                pgs += npgs

        if isinstance(pgs, dict):
            pgs = collect_grads(pgs, param_names)

        p_g.append(pgs)


def collect_grads(grads_dict, param_names, avg_pool=False):
    g = []
    for param_name in param_names:
        grad = grads_dict[param_name]
        grad = np.asarray(grad)
        shape = grad.shape

        if len(shape) == 1:
            continue

        grad = np.abs(grad)
        if len(shape) == 4:
            if shape[0] * shape[1] > 5000:
                continue
            grad = grad.reshape(shape[0], shape[1], -1)

        if len(shape) > 2 or shape[0] * shape[1] > 5000:
            if avg_pool:
                grad = np.mean(grad, -1)
            else:
                grad = np.max(grad, -1)

        g.append(grad.flatten())

    g = np.concatenate(g)
    return g


def aggregate_dicts(dicts, param_names):
    aggr_dict = dicts[0]

    for key in aggr_dict:
        aggr_dict[key] = np.asarray(aggr_dict[key])

    for d in dicts[1:]:
        for key in aggr_dict:
            aggr_dict[key] += np.asarray(d[key])

    return collect_grads(aggr_dict, param_names)


def train_multi_task_ps(data, num_iteration=6000, train_size=0.3, victim_id=0, seed=12345, warm_up_iters=100,
                        input_shape=(None, 3, 50, 50), n_workers=2, lr=0.01, attacker_id=1, filename="data",
                        p_prop=0.5, alpha_B=0., victim_all_nonprop=True, balance=False, k=5):

    splitted_X, splitted_y, X_test, y_test = prepare_data_biased(data, train_size, n_workers, seed=seed,
                                                                 victim_all_nonprop=victim_all_nonprop,
                                                                 p_prop=p_prop, balance=balance)
    p_test = y_test[:, 1]
    y_test = y_test[:, 0]

    classes = len(np.unique(y_test))
    # build test network
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network_dict = build_cnn(input_var, classes=classes, input_shape=input_shape)
    network = network_dict['fc2']
    prediction = lasagne.layers.get_output(network)
    prediction = lasagne.nonlinearities.softmax(prediction)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    global_params = lasagne.layers.get_all_params(network, trainable=True)
    global_grads = T.grad(loss, global_params)

    p_idx = 0
    grads_dict = dict()
    params_names = []
    for p, g in zip(global_params, global_grads):
        key = p.name + str(p_idx)
        params_names.append(key)
        p_idx += 1
        grads_dict[key] = g

    global_grad_fn = theano.function([input_var, target_var], grads_dict)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # build local workers
    worker_params = []
    worker_grad_fns = []
    data_gens = []

    for i in range(n_workers):
        if i == attacker_id:
            split_y = splitted_y[i]
            p, f, b_params, b_grad_fn, pval_fn = build_worker_attacker(input_shape, classes=classes, alph_B=alpha_B,
                                                                       infer_classes=len(np.unique(split_y[:, 1])))
            data_gen = inf_data(splitted_X[i], split_y[:, 0], y_b=split_y[:, 1], batchsize=32, shuffle=True)
            print 'Participant {} with {} data'.format(i, len(splitted_X[i]))
            data_gens.append(data_gen)
        elif i == victim_id:
            p, f = build_worker(input_shape, classes=classes)
            vic_X = np.vstack([splitted_X[i][0], splitted_X[i][1]])
            vic_y = np.concatenate([splitted_y[i][0][:, 0], splitted_y[i][1][:, 0]])
            vic_p = np.concatenate([splitted_y[i][0][:, 1], splitted_y[i][1][:, 1]])
            data_gen = inf_data(vic_X, vic_y, y_b=vic_p, batchsize=32, shuffle=True)

            data_gen_p = inf_data(splitted_X[i][0], splitted_y[i][0][:, 0], batchsize=32, shuffle=True)
            data_gen_np = inf_data(splitted_X[i][1], splitted_y[i][1][:, 0], batchsize=32, shuffle=True)

            data_gens.append(data_gen)
            print 'Participant {} with {} data'.format(i, len(splitted_X[i][0]) + len(splitted_X[i][1]))
        else:
            p, f = build_worker(input_shape, classes=classes)
            data_gen = inf_data(splitted_X[i], splitted_y[i][:, 0], batchsize=32, shuffle=True)
            print 'Participant {} with {} data'.format(i, len(splitted_X[i]))
            data_gens.append(data_gen)

        worker_params.append(p)
        worker_grad_fns.append(f)

    set_local(global_params, worker_params)

    train_pg, train_npg = [], []
    test_pg, test_npg = [], []
    X, y, _ = data

    # attacker's aux data
    X_adv, y_adv = splitted_X[attacker_id], splitted_y[attacker_id]
    p_adv = y_adv[:, 1]
    y_adv = y_adv[:, 0]

    indices = np.arange(len(X_adv))
    prop_indices = indices[p_adv == 1]
    nonprop_indices = indices[p_adv == 0]
    adv_gen = mix_inf_data(X_adv[prop_indices], splitted_y[attacker_id][prop_indices],
                           X_adv[nonprop_indices], splitted_y[attacker_id][nonprop_indices], batchsize=32, mix_p=0.2)

    X_adv = np.vstack([X_adv, X_test])
    y_adv = np.concatenate([y_adv, y_test])
    p_adv = np.concatenate([p_adv, p_test])

    indices = np.arange(len(p_adv))
    train_prop_indices = indices[p_adv == 1]
    train_prop_gen = inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices], 32, shuffle=True)

    indices = np.arange(len(p_test))
    nonprop_indices = indices[p_test == 0]
    n_nonprop = len(nonprop_indices)

    print 'Attacker prop data {}, non prop data {}'.format(len(train_prop_indices), n_nonprop)
    train_nonprop_gen = inf_data(X_test[nonprop_indices], y_test[nonprop_indices], 32, shuffle=True)

    train_mix_gens = []
    for train_mix_p in [0.4, 0.6, 0.8]:
        train_mix_gen = mix_inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices],
                                     X_test[nonprop_indices], y_test[nonprop_indices], batchsize=32, mix_p=train_mix_p)
        train_mix_gens.append(train_mix_gen)

    start_time = time.time()
    for it in range(num_iteration):
        aggr_grad = []
        set_local(global_params, worker_params)

        for i in range(n_workers):
            grad_fn = worker_grad_fns[i]
            data_gen = data_gens[i]
            if i == attacker_id:
                batch = next(adv_gen)
                inputs, targets = batch
                targetsB = targets[:, 1]
                targets = targets[:, 0]
                grads_dict = grad_fn(inputs, targets, targetsB)
            elif i == victim_id:
                if it % k == 0:
                    inputs, targets = next(data_gen_p)
                else:
                    inputs, targets = next(data_gen_np)
                grads_dict = grad_fn(inputs, targets)
            else:
                inputs, targets = next(data_gen)
                grads_dict = grad_fn(inputs, targets)

            if i != attacker_id:
                aggr_grad.append(grads_dict)

            grads = [grads_dict[name] for name in params_names]
            update_global(global_params, grads, lr)

        if it >= warm_up_iters:
            test_gs = aggregate_dicts(aggr_grad, param_names=params_names)
            if it % k == 0:
                test_pg.append(test_gs)
            else:
                test_npg.append(test_gs)

            if n_workers > 2:
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen_multi(train_mix_gen, train_nonprop_gen, train_pg, global_grad_fn,
                                                   iters=2, n_workers=n_workers, param_names=params_names)
                gradient_getter_with_gen_multi(train_prop_gen, train_nonprop_gen, train_pg, global_grad_fn,
                                               iters=2, n_workers=n_workers, param_names=params_names)
                gradient_getter_with_gen_multi(train_nonprop_gen, train_nonprop_gen, train_npg, global_grad_fn,
                                               iters=8, n_workers=n_workers, param_names=params_names)
            else:
                gradient_getter_with_gen(train_prop_gen, train_pg, global_grad_fn, iters=2,
                                         param_names=params_names)
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen(train_mix_gen, train_pg, global_grad_fn, iters=2,
                                             param_names=params_names)

                gradient_getter_with_gen(train_nonprop_gen, train_npg, global_grad_fn, iters=8,
                                         param_names=params_names)

        if (it + 1) % 500 == 0 and it > 0:
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            val_perr = 0
            val_pacc = 0
            val_pbatches = 0

            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            for batch in iterate_minibatches(X_test, p_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = pval_fn(inputs, targets)
                val_perr += err
                val_pacc += acc
                val_pbatches += 1

            sys.stderr.write("Iteration {} of {} took {:.3f}s\n".format(it + 1, num_iteration,
                                                                        time.time() - start_time))
            sys.stderr.write("  test accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches / 500 * 100))
            sys.stderr.write("  p-test accuracy:\t\t{:.2f} %\n".format(val_pacc / val_pbatches / 500 * 100))
            start_time = time.time()

    np.savez(SAVE_DIR + "{}.npz".format(filename),
             train_pg=train_pg, train_npg=train_npg, test_pg=test_pg, test_npg=test_npg)


if __name__ == '__main__':
    train_lfw()
