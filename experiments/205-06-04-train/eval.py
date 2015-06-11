#!/usr/bin/env python2

from __future__ import division

import sys

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import omnitune
from omnitune import skelcl

from omnitune.skelcl import ml
from omnitune.skelcl import db as _db


def skelcl_eval(classifier, testing, db):
    def eval_prediction(instance, label):
        # Create a set of (key,val) pairs.
        keys = [attr.name for attr in testing.attributes()]
        values = [value for value in instance]

        where = " AND ".join(["{0}=?".format(key) for key in keys])

        oracle = db.execute("SELECT runtime FROM features_runtime_stats "
                            "WHERE " + where, values).fetchone()
        values[-1] = label
        predicted = db.execute("SELECT runtime FROM features_runtime_stats "
                               "WHERE " + where, values).fetchone()

        if oracle is not None and predicted is not None:
            oracle, predicted = oracle[0], predicted[0]
            io.debug(oracle, predicted, oracle / predicted)
            return oracle / predicted
        else:
            io.debug("naughty")

    predictions = ml.evaluate(classifier, testing)
    return [eval_prediction(instance, label) for instance,label in
            zip(testing, predictions)]


def main():
    """
    Generate a test dataset.
    """
    training = ml.load_arff("oracle.arff")
    db = _db.MLDatabase(experiment.ORACLE_PATH)

    j48 = ml.create_classifier(training, "weka.classifiers.trees.J48",
                               "-C", "0.3")

    performance = skelcl_eval(j48, training, db)
    io.debug(performance)

if __name__ == "__main__":
    main()
