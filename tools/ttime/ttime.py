#!/usr/bin/env python3

import sys
from optparse import OptionParser

def version_and_quit(*data):
    print("ttimer version 0.0.1")
    print("Copyright (c) 2014 Chris Cummins")
    sys.exit(0)

class TTimeParser(OptionParser):
    def __init__(self):
        OptionParser.__init__(self)

        # Allow overriding the default handlers:
        self.set_conflict_handler("resolve")

        self.add_option("--version", action="callback",
                        callback=version_and_quit)
        self.add_option("-s", "--source", action="store", type="string",
                        dest="source", default="self")
        self.add_option("-n", "--number", action="store", type="int",
                        dest="number", default=30)

class Timer:
    def __init__(self, command, options):
        self.options = options
        self.command = command

    def run(self):
        print("Command:", self.command)
        print("Options:", self.options)

def main(argc, argv):
    # Get arguments from command line:
    parser = TTimeParser()
    (options, args) = parser.parse_args()

    # Run timer:
    t = Timer(args, options)
    t.run()

    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
