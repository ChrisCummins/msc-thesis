# Tunable knobs for SkelCL

For each, identify the type of property and the range of values.

1. Stencil implementation: {MapOverlap, Stencil}. There are two
   competing implementations with (near) identical interfaces.
1. Target execution device: {CPU,GPU}. Possibly mixed?
1. Data distribution: {single,copy,block,overlap}. Applicable for
   multi-GPU execution.
1. Mapping of data elements to threads: NUMERICAL. SkelCL assumes a
   one to one mapping of data elements to work items. This may not
   always be profitable.
1. TODO: Data locality?
