#!/usr/bin/env python2
from train import setup,run_real_benchmarks

def main():
    setup()
    while True:
        run_real_benchmarks()


if __name__ == "__main__":
    main()
