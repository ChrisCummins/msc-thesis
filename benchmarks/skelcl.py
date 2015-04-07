# skelcl.py - SkelCL benchmarking extensions.
#
from re import search
from subprocess import check_output

import os

from benchlib import *
from config import *

### CONFIG
SKELCL = path(CWD, '../skelcl')
SKELCL_BUILD = path(SKELCL, 'build')

DATA = path(CWD, 'data')
IMG = path(DATA, 'img')

# Get the current SkelCL git version.
def skelcl_version():
    return check_output(['git', 'rev-parse', 'HEAD']).strip()

class DeviceTypeArg(Argument):
    def __init__(self, type):
        Argument.__init__(self, "Device type",
                          "--device-type {type}".format(type=type))

class DeviceCountArg(Argument):
    def __init__(self, count):
        Argument.__init__(self, "Device count",
                          "--device-count {count}".format(count=count))

class IterationsArg(Argument):
    def __init__(self, i):
        Argument.__init__(self, "Iterations", "-i {i}".format(i=i))

class MapOverlapArg(Argument):
    def __init__(self):
        Argument.__init__(self, "MapOverlap", "--map-overlap")

class SkelCLHost(OpenCLHost):
    def devargs(self):
        args = []
        for i in range(1, len(self.GPUS) + 1):
            args.append([DeviceTypeArg("GPU"), DeviceCountArg(i)])

        if self.OPENCL_CPU:
            args.append([DeviceTypeArg("CPU")])

        return args

    @staticmethod
    def create(name):
        _hosts = {
            "florence": SkelCLHost("florence",
                                   cpu="Intel i5-2430M",
                                   mem=8,
                                   opencl_cpu=True),
            "cec": SkelCLHost("cec",
                              cpu="Intel i5-4570",
                              mem=8,
                              opencl_cpu=True),
            "dhcp-90-060": SkelCLHost("dhcp-90-060",
                                      cpu="Intel i7-2600K",
                                      mem=16,
                                      gpus=["NVIDIA GTX 690"],
                                      opencl_cpu=False),
            "whz5": SkelCLHost("whz5",
                               cpu="Intel i7-4770",
                               mem=16,
                               gpus=["NVIDIA GTX TITAN"],
                               opencl_cpu=False),
            "tim": SkelCLHost("tim",
                              cpu="Intel i7-2600K",
                              mem=8,
                              gpus=["NVIDIA GTX 590", "NVIDIA GTX 590",
                                    "NVIDIA GTX 590", "NVIDIA GTX 590"],
                              opencl_cpu=False),
            "monza": SkelCLHost("monza",
                                cpu="Intel i7-3820",
                                mem=8,
                                gpus=["AMD Tahiti 7970", "AMD Tahiti 7970"],
                                opencl_cpu=True)
        }
        return _hosts[name]

#
class SkelCLSourceTree(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "SkelCL version")

    def post(self, **kwargs):
        cd(SKELCL)
        output = check_output(["git", "rev-parse", "HEAD"])
        self.val = output.decode('utf-8').rstrip()

#
class InitTime(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "InitTime")

    def post(self, **kwargs):
        for line in kwargs['output']:
            match = search('skelcl::init\(\) time ([0-9]+) ms$', line)
            if match:
                self.val = int(match.group(1))
                return
        raise LookupError(kwargs)

#
class PrepareTimes(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "PrepareTimes")

    def post(self, **kwargs):
        self.val = {}
        for line in kwargs['output']:
            match = search('PROF\] ([a-zA-Z\(<>, \)]+)\[0x([0-9a-f]+)\]'
                           ' prepare ([0-9\.]+) ms',
                           line)
            if match:
                type = match.group(1)
                address = match.group(2)
                time = int(match.group(3))

                if type not in self.val:
                    self.val[type] = {}
                if address not in self.val[type]:
                    self.val[type][address] = []
                self.val[type][address].append(time)

#
class SwapTimes(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "SwapTimes")

    def post(self, **kwargs):
        self.val = []
        for line in kwargs['output']:
            match = search('PROF\] swap ?([0-9\.]+) ms', line)
            if match:
                time = float(match.group(1))
                self.val.append(time)

#
class ElapsedTimes(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "ElapsedTimes")

    def post(self, **kwargs):
        r = []
        for line in kwargs['output']:
            match = search('^Elapsed time:\s+([0-9]+)\s+', line)
            if match:
                r.append(int(match.group(1)))

        # Clamp each time at >= 1 ms. This is necessary for computations
        # that require less than .5 ms, which will be rounded down to 0.
        self.val = [max(x, 1) for x in r]

#
class SkeletonEventTimes(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "SkeletonEventTimes")

    def post(self, **kwargs):
        self.val = {}
        for line in kwargs['output']:
            # Parse profiling information.
            match = search('PROF\] ([a-zA-Z\(<>, \)]+)\[0x([0-9a-f]+)\]'
                           '\[([0-9]+)\] ([0-9\.]+) ms',
                           line)
            if match:
                type = match.group(1)
                address = match.group(2)
                id = int(match.group(3))
                run = float(match.group(4))

                # Record profiling information.
                if type not in self.val:
                    self.val[type] = {}
                if address not in self.val[type]:
                    self.val[type][address] = []
                self.val[type][address].append(run)

#
class ContainerEventTimes(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "ContainerEventTimes")

    def post(self, **kwargs):
        self.val = {}
        for line in kwargs['output']:
            # Parse profiling information.
            match = search('PROF\] ([a-zA-Z\(<>, \)]+)\[0x([0-9a-f]+)\]'
                           '\[([0-9]+)\] (ul|dl) ([0-9\.]+) ms', line)
            if match:
                type = match.group(1)
                address = match.group(2)
                id = int(match.group(3))
                direction = match.group(4)
                run = float(match.group(5))

                # Record profiling information.
                if type not in self.val:
                    self.val[type] = {}
                if address not in self.val[type]:
                    self.val[type][address] = {'ul': {}, 'dl': {}}
                self.val[type][address][direction][id] = run

#
class ProgramBuildTimes(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "ProgramBuildTimes")

    def post(self, **kwargs):
        self.val = []
        for line in kwargs['output']:
            match = search('PROF\] skelcl::Program::build\(\) ([0-9]+) ms$', line)
            if match:
                self.val.append(int(match.group(1)))

#
class Devices(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "Devices")

    def post(self, **kwargs):
        self.val = []
        for line in kwargs['output']:
            match = search('INFO\] Using device \`([^\']+)\' with', line)
            if match:
                self.val.append(match.group(1))

#
class DeviceCount(DerivedVariable):
    def __init__(self, **kwargs):
        DerivedVariable.__init__(self, "DeviceCount")

    def post(self, **kwargs):
        self.val = []
        for line in kwargs['output']:
            match = search('INFO\] Using ([0-9]+) OpenCL device\(s\) in total$', line)
            if match:
                self.val = int(match.group(1))
                return

#
class SkelCLBenchmark(Benchmark):
    def __init__(self, name):
        # Binary directory:
        self.dir = path(SKELCL_BUILD, 'examples', name)
        # Source directory:
        self.src = path(SKELCL, 'examples', name)
        # Path to binary:
        binpath = path(self.dir, name)

        # Superclass:
        Benchmark.__init__(self, name, binpath)

    def run(self, *args):
        cd(self.dir)
        return Benchmark.run(self, *args)

    def build(self):
        cd(self.dir)
        system(['make', self.name])

#
class SkelCLTestCase(TestCase):
    def __init__(self, benchmark,
                 host=LocalHost(),
                 invars=[],
                 outvars=[],
                 coutvars=set()):
        # Default variables.
        ins = []
        outs = [
            ElapsedTimes,
            InitTime,
            ProgramBuildTimes,
            PrepareTimes,
            SwapTimes,
            Devices,
            DeviceCount,
            SkeletonEventTimes,
            ContainerEventTimes
        ]
        couts = {SkelCLSourceTree}

        TestCase.__init__(self, benchmark,
                          host=host,
                          invars=ins + invars,
                          outvars=outs + outvars,
                          coutvars=couts.union(coutvars))

class StencilKnob(Knob):
    defheader = path(SKELCL, 'include/SkelCL/detail/StencilDef.h')

class StencilLocalSizeR(StencilKnob):
    DEFAULT = 32

    def __init__(self, val):
        Knob.__init__(self, "StencilLocalSizeR", val)

    def set(self, **kwargs):
        os.system("sed -r -i 's/(define KNOB_R) [0-9]+/\\1 {val}/' {path}"
                  .format(val=self.val, path=self.defheader))

class StencilLocalSizeC(StencilKnob):
    DEFAULT = 4

    def __init__(self, val):
        Knob.__init__(self, "StencilLocalSizeC", val)

    def set(self, **kwargs):
        os.system("sed -r -i 's/(define KNOB_C) [0-9]+/\\1 {val}/' {path}"
                  .format(val=self.val, path=self.defheader))

class AllPairsKnob(Knob):
    defheader = path(SKELCL, 'include/SkelCL/detail/AllPairsDef.h')

class AllPairsC(AllPairsKnob):
    DEFAULT = 32

    def __init__(self, val):
        AllPairsKnob.__init__(self, "AllPairsC", val)

    def set(self, **kwargs):
        os.system("sed -r -i 's/(define KNOB_C) [0-9]+/\\1 {val}/' {path}"
                  .format(val=self.val, path=self.defheader))

class AllPairsR(AllPairsKnob):
    DEFAULT = 8

    def __init__(self, val):
        AllPairsKnob.__init__(self, "AllPairsR", val)

    def set(self, **kwargs):
        os.system("sed -r -i 's/(define KNOB_R) [0-9]+/\\1 {val}/' {path}"
                  .format(val=self.val, path=self.defheader))

class AllPairsS(AllPairsKnob):
    DEFAULT = 16

    def __init__(self, val):
        AllPairsKnob.__init__(self, "AllPairsS", val)

    def set(self, **kwargs):
        os.system("sed -r -i 's/(define KNOB_S) [0-9]+/\\1 {val}/' {path}"
                  .format(val=self.val, path=self.defheader))

class AllPairsCRS(AllPairsKnob):
    DEFAULT = [
        AllPairsC.DEFAULT,
        AllPairsR.DEFAULT,
        AllPairsS.DEFAULT
    ]

    def __init__(self, val):
        AllPairsKnob.__init__(self, "AllPairsCRS", val)
        self.knobs = [
            AllPairsC(val[0]),
            AllPairsR(val[1]),
            AllPairsS(val[2])
        ]

    def set(self, **kwargs):
        [x.set(**kwargs) for x in self.knobs]


### VARIABLES

hosts = [
    SkelCLHost("florence",
               cpu="Intel i5-2430M",
               mem=8,
               opencl_cpu=True),
    SkelCLHost("cec",
               cpu="Intel i5-4570",
               mem=8,
               opencl_cpu=True),
    SkelCLHost("dhcp-90-060",
               cpu="Intel i7-2600K",
               mem=16,
               gpus=["NVIDIA GTX 690"],
               opencl_cpu=False),
    SkelCLHost("whz5",
               cpu="Intel i7-4770",
               mem=16,
               gpus=["NVIDIA GTX TITAN"],
               opencl_cpu=False),
    SkelCLHost("tim",
               cpu="Intel i7-2600K",
               mem=8,
               gpus=["NVIDIA GTX 590", "NVIDIA GTX 590",
                     "NVIDIA GTX 590", "NVIDIA GTX 590"],
               opencl_cpu=False),
    SkelCLHost("monza",
               cpu="Intel i7-3820",
               mem=8,
               gpus=["AMD Tahiti 7970", "AMD Tahiti 7970"],
               opencl_cpu=True)
]

benchmarks = [
    SkelCLBenchmark("CannyEdgeDetection"),
    SkelCLBenchmark("DotProduct"),
    SkelCLBenchmark("FDTD"),
    SkelCLBenchmark("GameOfLife"),
    SkelCLBenchmark("GaussianBlur"),
    SkelCLBenchmark("HeatEquation"),
    SkelCLBenchmark("MandelbrotSet"),
    SkelCLBenchmark("MatrixMultiply"),
    SkelCLBenchmark("SAXPY")
]

def masterhost():
    return hostname() in MASTER_HOSTS

def enumerateHarnesses(e, instantiate):
    harnesses = []

    _hosts = e["hosts"]
    _benchmarks = e["benchmarks"]
    _knobs = e["knobs"]

    for _host, _benchmark in product(_hosts, _benchmarks):
        _args = _benchmarks[_benchmark]["args"]
        _args.update(e["args"])

        _vals = [[[arg, val] for val in _args[arg]] for arg in _args]
        argpermutations = [[Argument(x[0], x[1]) for x in args]
                           for args in list(product(*_vals))]


        _vals = list(product(*[_knobs[x] for x in _knobs]))
        knobpermutations = [[x(y) for x,y in zip(_knobs, k)] for k in _vals]

        host = SkelCLHost.create(_host)
        benchmark = SkelCLBenchmark(_benchmark)

        for args, knobs in product(argpermutations, knobpermutations):
            harnesses += instantiate(host, benchmark, args, knobs)

    return harnesses


#
def gettimes(samples):
    inittimes = []
    buildtimes = []
    preptimes = []
    ultimes = []
    skeltimes = []
    swaptimes = []
    dltimes = []

    def parsesample(sample):
        it = lookup1(sample, InitTime)
        bt = lookup1(sample, ProgramBuildTimes)
        pt = lookup1(sample, PrepareTimes)
        swt = lookup1(sample, SwapTimes)
        st = lookup(sample, SkeletonEventTimes)
        ct = lookup(sample, ContainerEventTimes)
        ndevices = len(lookup1(sample, Devices).val)

        inittimes.append(it.val)
        buildtimes.append(sum(bt.val))
        swaptimes.append(sum(swt.val))

        for type in pt.val:
            for address in pt.val[type]:
                preptimes.append(sum(pt.val[type][address]))

        # Collect skeleton and container OpenCL event times. Note here
        # that we are first summing up the total times for *all*
        # events of each type, and that each event time is divided by
        # the number of devices.

        # Skeleton times
        for var in st:
            val = var.val
            for type in val:
                for address in val[type]:
                    skeltimes.append(sum(val[type][address]) / ndevices)

        # Container upload and download times.
        for var in ct:
            val = var.val
            for type in val:
                for address in val[type]:
                    for direction in val[type][address]:
                        times = [val[type][address][direction][x]
                                 for x in val[type][address][direction]]
                        if direction == "ul":
                            ultimes.append(sum(times) / ndevices)
                        elif direction == "dl":
                            dltimes.append(sum(times) / ndevices)
                        else:
                            raise Exception("Unknown direction!", direction)

    [parsesample(x) for x in samples]
    return inittimes, buildtimes, preptimes, ultimes, skeltimes, swaptimes, dltimes

# Filter a list of invars to return arguments which determine the
# execution device. E.g. DeviceTypeArg, and DeviceCountArg.
def getdeviceargs(invars):
    args = [lookup1(invars, DeviceTypeArg)]

    if not search("CPU", args[0].val):
        args.append(lookup1(invars, DeviceCountArg))

    return args
