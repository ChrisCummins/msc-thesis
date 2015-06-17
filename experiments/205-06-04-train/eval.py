#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

from functools import partial
from itertools import product

import numpy as np
import scipy
from scipy import stats

import weka
from weka.classifiers import Classifier as WekaClassifier
from weka.classifiers import FilteredClassifier as WekaFilteredClassifier
from weka.filters import Filter as WekaFilter
from weka.core.classes import Random as WekaRandom
from weka.core.dataset import Instances

import labm8 as lab
from labm8 import fmt
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml
from labm8.db import where

import omnitune
from omnitune.skelcl import db as _db
from omnitune.skelcl import hash_params
from omnitune.skelcl import unhash_params

import experiment


def summarise_perfs(perfs):
    def _fmt(n):
        return "{:.3f}".format(n)

    print("        n: ", len(perfs))
    print("     mean: ", _fmt(labmath.mean(perfs)))
    print("  geomean: ", _fmt(labmath.geomean(perfs)))
    print("      min: ", _fmt(min(perfs)))
    print("      max: ", _fmt(max(perfs)))
    print()


def oracle_params_arff(db):
    nominals = [
        49,  # dev_double_fp_config
        50,  # dev_endian_little
        51,  # dev_execution_capabilities
        52,  # dev_extensions
        54,  # dev_global_mem_cache_type
        57,  # dev_host_unified_memory
        63,  # dev_image_support
        65,  # dev_local_mem_type
        96,  # dev_queue_properties
        97,  # dev_single_fp_config
        98,  # dev_type
        100, # dev_vendor_id
    ]
    force_nominal_args = ["-N", ",".join([str(index) for index in nominals])]
    data = ml.load_csv("/tmp/omnitune/csv/oracle_params.csv",
                       options=force_nominal_args)
    data.class_is_last()

    return data


def perf_fn(db, one_r, instance, predicted, oracle):
    scenario = instance.get_string_value(0)
    speedup = db.speedup(scenario, one_r, predicted)

    if predicted == oracle:
        return 1, speedup
    else:
        return db.perf(scenario, predicted), speedup


def one_r_fn(db, one_r, *args, **kwargs):
    return one_r


def random_fn(db, instance, max_wgsize, wg_c, wg_r):
    """
    Random workgroup size callback.

    Pick a random workgroup size from the parameters table which is
    smaller than or equal to the max workgroup size.
    """
    return db.execute("SELECT wg_c,wg_r\n"
                      "FROM params\n"
                      "WHERE wg_c * wg_r <= ?\n"
                      "ORDER BY RANDOM()\n"
                      "LIMIT 1", (max_wgsize,)).fetchone()


def reshape_fn(db, instance, max_wgsize, wg_c, wg_r):
    """
    Reshape callback.

    Reduce the
    """
    all_wg_c = db.wg_c
    all_wg_r = db.wg_r

    c = all_wg_c.index(wg_c)
    r = all_wg_r.index(wg_r)
    i = 0

    while all_wg_c[c] * all_wg_r[r] > max_wgsize:
        if c > 1: c -= 1
        if r > 1: r -= 1
        i += 1
        if i >= 100:
            raise Error("Failed to shrink wgsize {0}x{1} below maximum size {2}"
                        .format(all_wg_c[c], all_wg_r[r], max_wgsize))

    return all_wg_c[c], all_wg_r[r]


def eval_instance(classifier, instance, perf_fn, err_fn):
    # Get oracle workgroup size.
    oracle = instance.get_string_value(instance.class_index)

    # Get predicted workgroup size.
    value_index = classifier.classify_instance(instance)
    class_attr = instance.dataset.attribute(instance.class_index)
    predicted = class_attr.value(value_index)

    if predicted == oracle:
        return 1, 0, perf_fn(instance, predicted, oracle)
    else:
        # Determine if predicted workgroup size is valid or not. A
        # valid predicted is one which is within the max_wgsize for
        # that particular instance.
        max_wgsize_attr = instance.dataset.attribute_by_name("kern_max_wg_size")
        max_wgsize_attr_index = max_wgsize_attr.index
        max_wgsize = instance.get_value(max_wgsize_attr_index)
        wg_c, wg_r = unhash_params(predicted)
        is_valid = wg_c * wg_r < max_wgsize

        if is_valid:
            return 0, 0, perf_fn(instance, predicted, oracle)
        else:
            new_wg_c, new_wg_r = err_fn(instance, max_wgsize, wg_c, wg_r)
            return 0, 1, perf_fn(instance, hash_params(new_wg_c, new_wg_r), oracle)


def eval_classifier(training, testing, classifier_class, *args, **kwargs):
    # Create attribute filer.
    rm = WekaFilter(classname="weka.filters.unsupervised.attribute.Remove")
    # Create classifier.
    classifier = WekaClassifier(classname=classifier_class)
    # Create meta-classifier.
    meta = WekaFilteredClassifier()
    meta.set_property("filter", rm)
    meta.set_property("classifier", classifier)

    # Train classifier.
    meta.build_classifier(training)

    results = [eval_instance(classifier, instance, *args, **kwargs)
               for instance in testing]

    correct, invalid, performance = zip(*results)
    perf_oracle, speedups = zip(*performance)

    total = len(correct)

    accuracy = (sum(correct) / total) * 100
    ratio_invalid = (sum(invalid) / total) * 100

    min_perf = min(perf_oracle) * 100
    mean_perf = labmath.geomean(perf_oracle) * 100
    max_perf = max(perf_oracle) * 100

    min_speedup = min(speedups)
    mean_speedup = labmath.geomean(speedups)
    max_speedup = max(speedups)

    return (accuracy, ratio_invalid, min_perf, mean_perf, max_perf,
            min_speedup, mean_speedup, max_speedup)

def xvalidate_classifier(dataset, nfolds, *args, **kwargs):
    seed = 1

    rnd = WekaRandom(seed)

    # Shuffle the dataset.
    dataset.randomize(rnd)

    num_instances = dataset.num_instances
    fold_size = labmath.ceil(num_instances / nfolds)

    data = []

    for i in range(nfolds):
        testing_start = i * fold_size
        testing_end = min(testing_start + fold_size, num_instances - 1)

        # Calculate dataset indices for testing and training data.
        testing_range = (testing_start, testing_end - testing_start)
        left_range = (0, testing_start)
        right_range = (testing_end, num_instances - testing_end)

        # If there's nothing to test, move on.
        if testing_range[1] < 1: continue

        # Create testing and training folds.
        testing = Instances.copy_instances(dataset, *testing_range)
        left = Instances.copy_instances(dataset, *left_range)
        right = Instances.copy_instances(dataset, *right_range)
        training = Instances.append_instances(left, right)

        # Test on folds.
        data.append(eval_classifier(training, testing, *args, **kwargs))

    transpose = map(list, zip(*data))

    return [nfolds] + [labmath.mean(row) for row in transpose]


def main():
    """
    Evaluate dataset and omnitune performance.
    """
    ml.start()

    db = _db.Database(experiment.ORACLE_PATH)

    nfolds = 10

    one_r = db.one_r()
    print("ONE R:", one_r[0])
    summarise_perfs(db.perf_param(one_r[0]).values())

    db.dump_csvs("/tmp/omnitune/csv")
    dataset = oracle_params_arff(db);

    classifiers = (
        ("ZeroR", "weka.classifiers.rules.ZeroR"),
        ("SVM", "weka.classifiers.functions.SMO"),
        ("Logistic", "weka.classifiers.functions.SimpleLogistic"),
        ("RandomForest", "weka.classifiers.trees.RandomForest"),
        ("NaiveBayes", "weka.classifiers.bayes.NaiveBayes"),
        ("J48", "weka.classifiers.trees.J48"),
    )

    err_fns = (
        ("one_r", partial(one_r_fn, db, unhash_params(one_r[0]))),
        ("random", partial(random_fn, db)),
        ("reshape", partial(reshape_fn, db))
    )

    perf_cb = partial(perf_fn, db, one_r[0])

    results = []
    for c, e in list(product(classifiers, err_fns)):
        classifier, err_fn = c[1], e[1]
        classifier_name, err_fn_name = c[0], e[0]
        data = xvalidate_classifier(dataset, nfolds, classifier,
                                    perf_cb, err_fn)
        results.append([classifier_name, err_fn_name] + data)

    print(fmt.table(results, columns=(
        "CLASSIFIER",
        "ERR_FN",
        "NFOLDS",
        "ACCURACY (%)",
        "INVALID (%)",
        "MIN_ORACLE (%)",
        "MEAN_ORACLE (%)",
        "MAX_ORACLE (%)",
        "SPEEDUP_MIN",
        "SPEEDUP_MEAN",
        "SPEEDUP_MAX"
    )))

    ml.stop()


if __name__ == "__main__":
    main()
