# weka.py - Interface functions for interacting with Weka.
#
#
from __future__ import print_function
from copy import copy
from re import sub
from sys import stdout

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
def _get_schema(data):
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
def mkarff(data, relation="data", file=stdout):
    print("@RELATION {0}".format(relation), file=file)
    print(file=file)

    # Determine the schema, and print.
    for attr in _get_schema(data):
        print("@ATTRIBUTE {0} {1}".format(attr.name, attr.type), file=file)
    print(file=file)

    # Print the dataset values.
    print("@DATA", file=file)
    for d in data:
        dd = [str(x.value) for x in d]
        print(','.join(dd), file=file)

    if file != stdout:
        print("Wrote '{0}'...".format(file.name))
