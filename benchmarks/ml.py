# weka.py - Interface functions for interacting with Weka.
#
#
from __future__ import print_function
import math

from copy import copy
from random import shuffle
from re import sub
from sys import stdout,exit
import json

import weka
import weka.core
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import Classifier

jvm.start()
_LOADER = Loader(classname="weka.core.converters.ArffLoader")


def load_arff(path):
    """
    Return a weka-wrapper Instances object for the dataset at "path".
    """
    arff = _LOADER.load_file(path)
    arff.class_is_last()
    return arff


def get_J48(training_data, *args, **kwargs):
    """
    Given an Instances object "training_data", build a J48 classifier.
    """
    tree = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
    tree.build_classifier(training_data)
    return tree


def get_wgsize(dataset, value_index):
    """
    Convert a value index into a wgsize value.
    """
    attr_index = dataset.num_attributes - 1
    return dataset.attribute(attr_index).value(value_index)

_DATA_JSON = json.load(open("./results/e14/data.arff.json"))["data"]

def match_result(result, d):
    for key in d:
        if key == "LocalSize":
            continue
        if d[key] != result[key]:
            try:
                if abs(float(d[key]) - float(result[key])) > 0.000001:
                    raise Exception
            except Exception:
                return False
    return True


def lookup_speedups(instances, instance, wgsize):
    a = []
    for attr in instances.attributes():
        a.append(attr.name)

    d = {}
    for attr,val in zip(a, instance):
        d[attr] = val

    match = None
    for result in _DATA_JSON:
        if match_result(result, d):
            match = result
            break

    if match is None:
        print("SHITTTTTTTTTTTTTTTTTTTTTT")
        print(d)
        exit(0)

    try:
        speedup = result["Speedup" + wgsize]
    except Exception:
        print("PREDICT BOOB")
        speedup = result["Speedup32x4"]
    try:
        oracle_speedup=result["Speedup" + result["OracleLocalSize"]]
    except Exception:
        print("ORACLE BOOB")
        oracle_speedup = result["Speedup32x4"]
    return speedup[0], oracle_speedup[0]

def evaluate_J48(training_data, testing_data, *args, **kwargs):
    """
    Return a list of speedups for a J48 classifier trained on
    "training_data" and tested using "testing_data".
    """
    tree = get_J48(training_data, *args, **kwargs)

    speedups, oracle_speedups = [], []
    for index, inst in enumerate(testing_data):
        pred = tree.classify_instance(inst)
        wgsize = get_wgsize(training_data, pred)
        speedup, oracle_speedup = lookup_speedups(testing_data, inst, wgsize)
        if speedup > 0:
            speedups.append(speedup)
            oracle_speedups.append(oracle_speedup)

    return speedups, oracle_speedups


def split_arff(path, k=10):
    """
    Split an arff file into "k" folds.
    """
    arff = [x.rstrip() for x in open(path).readlines()]
    header, rows = [], []
    for i in range(len(arff)):
        line = arff[i]
        header.append(line)
        if line == "@DATA":
            break

    linenum = i+1

    for i in range(linenum, len(arff)):
        line = arff[i]
        rows.append(line)

    nrows = len(rows)
    splitsize = int(math.ceil(nrows / float(k)))

    training_paths, validation_paths = [], []

    # Shuffle data randomly.
    shuffle(rows)

    j = 0
    for i in range(0, len(rows), splitsize):
        j += 1
        training_path = path + "-{0:02d}-training".format(j)
        validation_path = path + "-{0:02d}-validation".format(j)

        validation_data = rows[i:i+splitsize]
        training_data = rows[:i] + rows[i+splitsize:]

        with open(validation_path, "w") as file:
             file.write("\n".join(header + validation_data))
        with open(training_path, "w") as file:
            file.write("\n".join(header + training_data))

        training_paths.append(training_path)
        validation_paths.append(validation_path)

    return training_paths, validation_paths


# Represents an (<attribute> <type> <value>) tuple. Performs automatic
# escaping of characters for the arff file format.
class ArffAttribute:
    ESCAPE_CHAR = "_"

    # Create a new ArffAttribute and escape all values.
    def __init__(self, name, type, value):
        self.type = str(type).upper()
        self.name = self._escape_str(name)
        self.value = self._escape_val(value)

    # Escape a string.
    def _escape_str(self, value):
        # Cast to string and discard whitespace.
        string = str(value).strip()
        lowerstring = string.lower()
        # Do not allow empty strings. Use "None" value.
        if not string:
            return "None"
        # Replace "yes/no" strings with "1/0".
        elif lowerstring == "yes":
            return "1"
        elif lowerstring == "no":
            return "0"
        # Escape characters with "_" underscores.
        else:
            return sub("[,\-\s\(\)]+", self.ESCAPE_CHAR, string)

    # Escape a value as either a number of a string, depending on the
    # attribute type.
    def _escape_val(self, value):
        if self.type == "NUMERIC":
            # Ensure a numeric value (defaults to 0).
            return value if value else 0
        else:
            return self._escape_str(value)

# Return the schema for a dataset.
def make_arff_schema(data):
    schema = [copy(attr) for attr in data[0]]

    # Iterate over each attribute in the schema. If the attribute is
    # nominal, set the type to be a set of all unique values.
    for i in range(len(schema)):
        if schema[i].type == "NOMINAL":
            # Get unique values for nominal set:
            uniq = set()
            [uniq.add(row[i].value) for row in data]
            # Create a type string of the form: {val1,...valn}
            schema[i].type = "{" + ",".join(uniq) + "}"

    return schema

# @data a list of ArffAtributes.
# @relation the name of the relation set.
# @file a file object to write output to.
def mkarff(data, relation="data", file=stdout, schema=None):
    print("@RELATION {0}".format(relation), file=file)
    print(file=file)

    # Calculate the schema, if needed.
    if schema == None:
        schema = make_arff_schema(data)

    # Print the schema.
    for attr in schema:
        print("@ATTRIBUTE {0} {1}".format(attr.name, attr.type), file=file)
    print(file=file)

    # Print the dataset values.
    print("@DATA", file=file)
    for d in data:
        dd = [str(x.value) for x in d]
        print(','.join(dd), file=file)

    if file != stdout:
        print("Wrote '{0}'...".format(file.name))
