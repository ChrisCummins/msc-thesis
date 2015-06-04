import itertools

import labm8
from labm8 import io
from labm8 import fs
from labm8 import system


ROOT = fs.path("~/src/msc-thesis")

EXAMPLES_BUILD = fs.path(ROOT, "skelcl/build/examples/")
EXAMPLES_SRC = fs.path(ROOT, "skelcl/examples/")

## Arguments

if system.HOSTNAME == "cec":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"]]
elif system.HOSTNAME == "dhcp-90-060":
    DEVARGS = [["--device-type", "GPU", "--device-count", "1"]]
elif system.HOSTNAME == "florence":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"]]
elif system.HOSTNAME == "monza":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "2"]]
elif system.HOSTNAME == "tim":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "2"],
               ["--device-type", "GPU", "--device-count", "3"],
               ["--device-type", "GPU", "--device-count", "4"]]
elif system.HOSTNAME == "whz5":
    DEVARGS = [["--device-type", "GPU", "--device-count", "1"]]
else:
    io.fatal("Unrecognised hostname!")
