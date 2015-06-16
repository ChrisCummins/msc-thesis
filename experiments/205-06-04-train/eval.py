#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import sys

import labm8 as lab
from labm8 import io
from labm8 import fs
from labm8 import math as labmath
from labm8 import ml

import omnitune
from omnitune.skelcl import db as _db

import experiment


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


# TODO: Unfinished code:
def eval_classifier(classifier, testing, db):
    def eval_prediction(instance, label):
        values = [value for value in instance]
        oracle = values[-1]
        return 1 if label == oracle else 0

        # # Create a set of (key,val) pairs.
        # keys = [attr.name for attr in testing.attributes()]
        # values = [value for value in instance]

        # where = " AND ".join(["{0}=?".format(key) for key in keys])

        # oracle = db.execute("SELECT runtime FROM features_runtime_stats "
        #                     "WHERE " + where, values).fetchone()
        # values[-1] = label
        # predicted = db.execute("SELECT runtime FROM features_runtime_stats "
        #                        "WHERE " + where, values).fetchone()

        # if oracle is not None and predicted is not None:
        #     oracle, predicted = oracle[0], predicted[0]
        #     io.debug(oracle, predicted, oracle / predicted)
        #     return oracle / predicted
        # else:
        #    io.debug("naughty")

    predictions = [classifier.classify(instance) for instance in testing]
    performance = [eval_prediction(instance, label) for instance,label in
                   zip(testing, predictions)]
    io.info("Performance on", len(predictions), "instances:",
            labmath.mean(performance))

def main():
    """
    Evaluate dataset and omnitune performance.
    """
    ml.start()

    db = _db.Database(experiment.ORACLE_PATH)

    io.debug("ZERO_R", db.zero_r())
    io.debug("ONE_R", db.one_r())

    db.dump_csvs("/tmp/omnitune/csv")

    dataset = oracle_params_arff(db);

    classifiers = {
        "J48": ml.J48(dataset),
        "NaiveBayes": ml.NaiveBayes(dataset),
    }

    for name,classifier in classifiers.iteritems():
        io.info(name)
        eval_classifier(classifier, dataset, db)

    ml.stop()


if __name__ == "__main__":
    main()
