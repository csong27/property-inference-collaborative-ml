import numpy as np

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from distributed_sgd import SAVE_DIR


def inference_attack(data, norm=True, scale=True):
    train_pg, train_npg, test_pg, test_npg = data

    train_pg = np.asarray(train_pg)
    train_npg = np.asarray(train_npg)
    test_pg = np.asarray(test_pg)
    test_npg = np.asarray(test_npg)
    print("train ps-nps {}-{} ** test ps-nps {}-{}".format(train_pg.shape, train_npg.shape, test_pg.shape,
                                                           test_npg.shape))

    X_train = np.vstack([train_pg, train_npg])
    y_train = np.concatenate([np.ones(len(train_pg)), np.zeros(len(train_npg))])

    X_test = np.vstack([test_pg, test_npg])
    y_test = np.concatenate([np.ones(len(test_pg)), np.zeros(len(test_npg))])

    X_train = np.abs(X_train)
    X_test = np.abs(X_test)

    if norm:
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=5, min_samples_leaf=5, min_samples_split=5)

    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    print classification_report(y_true=y_test, y_pred=y_pred)
    print 'AUC: ', roc_auc_score(y_true=y_test, y_score=y_score)


def evaluate_lfw(task='gender', attr="race", prop_id=2, n_workers=2, k=5, alpha_B=0.):
    filename = "lfw_psMT_{}_{}_{}_alpha{}_k{}".format(task, attr, prop_id, alpha_B, k)

    if n_workers > 2:
        filename += '_n{}'.format(n_workers)

    with np.load(SAVE_DIR + '{}.npz'.format(filename)) as f:
        train_pg, train_npg, test_pg, test_npg = f['train_pg'], f['train_npg'], f['test_pg'], f['test_npg']
    inference_attack((train_pg, train_npg, test_pg, test_npg))


if __name__ == '__main__':
    evaluate_lfw()
