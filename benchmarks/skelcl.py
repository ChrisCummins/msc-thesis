from benchlib import *

class DeviceTypeArg(Argument):
    def __init__(self, type):
        Argument.__init__(self, "Device type",
                          "--device-type {type}".format(type=type))

class DeviceCountArg(Argument):
    def __init__(self, count):
        Argument.__init__(self, "Device count",
                          "--device-count {count}".format(count=count))

class SkelCLHost(OpenCLHost):
    def devargs(self):
        args = []
        for i in range(1, len(self.GPUS) + 1):
            args.append([DeviceTypeArg("GPU"), DeviceCountArg(i)])

        if self.OPENCL_CPU:
            args.append([DeviceTypeArg("CPU")])

        return args

#
class SkelCLElapsedTimes(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Elapsed times")

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
class SkelCLSourceTree(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "SkelCL version")

    def post(self, **kwargs):
        cd(SKELCL)
        output = check_output(["git", "rev-parse", "HEAD"])
        self.val = output.decode('utf-8').rstrip()


#
class SkeletonEventTimes(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Skeleton Event timings")

    def post(self, **kwargs):
        self.val = {}
        for line in kwargs['output']:
            # Parse profiling information.
            match = search('PROF\] ([a-zA-Z\(<>\)]+) 0x([0-9a-f]+), '
                           'clEvent: ([0-9]+), ([a-zA-Z]+): ([0-9\.]+) ms',
                           line)
            if match:
                address = match.group(2)
                name = match.group(1)
                id = int(match.group(3))
                type = match.group(4)
                time = float(match.group(5))

                # Record profiling information.
                if address not in self.val:
                    self.val[address] = {}
                    self.val[address]['events'] = []
                while len(self.val[address]['events']) <= id:
                    self.val[address]['events'].append({})
                self.val[address]['name'] = name
                self.val[address]['events'][id][type] = time

#
class ContainerEventTimes(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Container Event timings")

    def post(self, **kwargs):
        self.val = {}
        for line in kwargs['output']:
            # Parse profiling information.
            match = search('PROF\] ([a-zA-Z\(<>\)]+) 0x([0-9a-f]+), '
                           'clEvent: ([0-9]+), (upload|download) '
                           '([a-zA-Z]+): ([0-9\.]+) ms', line)
            if match:
                address = match.group(2)
                name = match.group(1)
                id = int(match.group(3))
                direction = match.group(4)
                type = match.group(5)
                time = match.group(6)

                # Record profiling information.
                if name not in self.val:
                    self.val[name] = {}
                    self.val[name][address] = {'upload': {}, 'download': {}}
                if id not in self.val[name][address][direction]:
                    self.val[name][address][direction][id] = {}
                self.val[name][address][direction][id][type] = time

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
            SkelCLElapsedTimes,
            SkeletonEventTimes,
            ContainerEventTimes
        ]
        couts = {SkelCLSourceTree}

        TestCase.__init__(self, benchmark,
                          host=host,
                          invars=ins + invars,
                          outvars=outs + outvars,
                          coutvars=couts.union(coutvars))

class StencilLocalSize(Knob):
    header = path(SKELCL, 'include/SkelCL/detail/StencilDef.h')

    def __init__(self, val):
        Knob.__init__(self, "StencilLocalSize", val)

    def set(self, **kwargs):
        r, c = self.val[0], self.val[1]
        os.system("sed -r -i 's/(define KNOB_R) [0-9]+/\\1 {val}/' {path}"
                  .format(val=r, path=self.header))
        os.system("sed -r -i 's/(define KNOB_C) [0-9]+/\\1 {val}/' {path}"
                  .format(val=c, path=self.header))

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
    SkelCLBenchmark("HeatSimulation"),
    SkelCLBenchmark("MandelbrotSet"),
    SkelCLBenchmark("MatrixMultiply"),
    SkelCLBenchmark("SAXPY")
]
