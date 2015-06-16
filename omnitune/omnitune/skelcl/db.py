from __future__ import division
from __future__ import print_function

import subprocess
import sqlite3 as sql

import labm8 as lab
from labm8 import db
from labm8 import fmt
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml

from labm8.db import placeholders
from labm8.db import where

import omnitune
from omnitune.skelcl import features

from . import get_kernel_name_and_type
from . import get_user_source
from . import hash_dataset
from . import hash_device
from . import hash_kernel
from . import hash_params
from . import hash_scenario

from space import ParamSpace

from pkg_resources import resource_string


def sql_command(name):
    """
    Return named SQL command.

    SQL commands may be stored in external files, located under:

        omnitune/skelcl/data/<command-name>.sql

    Arguments:

        name (str): Name of the SQL command.

    Returns:

        str: The SQL command.
    """
    return resource_string(__name__, "data/" + name + ".sql")


class Database(db.Database):
    """
    Persistent database store for Omnitune SkelCL data.
    """

    def __init__(self, path=fs.path(omnitune.LOCAL_DIR, "skelcl.db")):
        """
        Create a new connection to database.

        Arguments:
           path (optional) If set, load database from path. If not, use
               standard system-wide default path.
        """
        super(Database, self).__init__(path)

        # Read SQL commands from files.
        self._insert_runtime_stat = sql_command("insert_runtime_stat")
        self._insert_oracle_param = sql_command("insert_oracle_param")
        self._select_perf_scenario = sql_command("select_perf_scenario")
        self._select_perf_param = sql_command("select_perf_param")

        # Get the database version.
        try:
            # Look up the version in the table.
            query = self.execute("SELECT version from version")
            self.version = query.fetchone()[0]
        except Exception:
            # Base case: This is pre-versioning.
            self.version = 0

    @property
    def params(self):
        return [row[0] for row in
                self.execute("SELECT id FROM params")]

    @property
    def wg_r(self):
        return [row[0] for row in
                self.execute("SELECT DISTINCT wg_r FROM params "
                             "ORDER BY wg_r ASC")]

    @property
    def wg_c(self):
        return [row[0] for row in
                self.execute("SELECT DISTINCT wg_c FROM params "
                             "ORDER BY wg_c ASC")]

    @property
    def devices(self):
        return [row[0] for row in
                self.execute("SELECT id FROM devices")]

    @property
    def gpus(self):
        return [row[0] for row in
                self.execute("SELECT id FROM devices where type=4")]

    @property
    def cpus(self):
        return [row[0] for row in
                self.execute("SELECT id FROM devices where type=2")]

    @property
    def datasets(self):
        return [row[0] for row in
                self.execute("SELECT id FROM datasets")]

    @property
    def kernels(self):
        return [row[0] for row in
                self.execute("SELECT id FROM kernels")]

    @property
    def kernel_names(self):
        return [row[0] for row in
                self.execute("SELECT DISTINCT name FROM kernel_names")]

    @property
    def num_scenarios(self):
        return self.execute("SELECT Count(*) from scenarios").fetchone()[0]

    @property
    def scenarios(self):
        return [row[0] for row in
                self.execute("SELECT id FROM scenarios")]

    @property
    def mean_samples(self):
        return self.execute("SELECT AVG(num_samples) FROM "
                            "runtime_stats").fetchone()[0]

    def run(self, name):
        """
        Run the names SQL script.

        Arguments:

            name (str): Name of the SQL script to be passed to
              sql_command()
        """
        self.executescript(sql_command(name))

    def runscript(self, name):
        """
        Run an sqlite subprocess, passing as input the named script.

        Arguments:

            name (str): Name of the SQL script to be passed to
              sql_command().

        Returns:

            (str, str): sqlite3 subprocess stdout and stderr.
        """
        script = sql_command(name)
        process = subprocess.Popen(["sqlite3", self.path],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        return process.communicate(input=script)

    def create_tables(self):
        """
        Setup the tables.

        Assumes tables do not already exist.
        """
        self.run("create_tables")

    def _id(self, table, lookup_columns, lookup_vals, feature_fn, id_fn):
        """
        Lookup the ID of a property.

        First, check the lookup tables. If there's no entry in the
        lookup tables, perform feature extraction, and cache results
        for future use.

        Arguments:

            table (str): The name of type to lookup. Table names are
              derived as {table}s for features, and {table}_lookup for
              feature lookup table.
            lookup_columns (list of str): Column names in the lookup table.
            lookup_vals (list): Column values in the lookup table.
            feature_fn (function): Function which accepts *lookup_vals
              as argument and returns a tuple of feature values.
            id_fn (function): Function which accepts a tuple of feature values
              and returns a unique ID.

        Returns:

            str: The unique ID.
        """
        lookup_table = table + "_lookup"
        features_table = table + "s"

        # Query lookup table.
        select = ("SELECT id FROM " + lookup_table + " WHERE " +
                  where(*lookup_columns))
        query = self.execute(select, lookup_vals).fetchone()

        # If there's an entry in the lookup table, return.
        if query:
            return query[0]

        # If there's no entry in lookup table, perform feature extraction.
        features = feature_fn(*lookup_vals)
        id = id_fn(features)

        io.debug("Added lookup table entry for", table, id, "...")

        # Add entry to features table. Since the relationship between
        # "lookup key" -> "feature id" is many -> one, there may
        # already be an entry in the features table.
        row = (id,) + features
        insert = ("INSERT OR IGNORE INTO " + features_table + " VALUES " +
                  placeholders(*row))
        self.execute(insert, row)

        # Add entry to lookup table.
        row = lookup_vals + (id,)
        insert = ("INSERT INTO " + lookup_table + " VALUES " +
                  placeholders("", *lookup_columns))
        self.execute(insert, row)

        self.commit()
        return id

    def device_id(self, name, count):
        """
        Lookup the ID of a device.

        Arguments:

            name (str): The name of the device.
            count (int): The number of devices.

        Returns:

            str: The unique device ID.
        """
        return self._id("device", ("name", "count"), (name, count),
                        features.device, hash_device)

    def kernel_id(self, north, south, east, west, max_wg_size, source):
        """
        Lookup the ID of a kernel.

        Arguments:

            north (int): The stencil shape north direction.
            south (int): The stencil shape south direction.
            east (int): The stencil shape east direction.
            west (int): The stencil shape west direction.
            max_wg_size (int): The maximum kernel workgroup size.
            source (str): The stencil kernel source code.

        Returns:

            str: The unique kernel ID.
        """
        return self._id("kernel",
                        ("north", "south", "east", "west", "max_wg_size", "source"),
                        (north,    south,   east,   west,   max_wg_size,   source),
                        features.kernel, hash_kernel)

    def datasets_id(self, width, height, tin, tout):
        """
        Lookup the ID of a dataset.

        Arguments:

            data_width (int): The number of columns of data.
            data_height (int): The number of rows of data.
            type_in (str): The input data type.
            type_out (str): The output data type.

        Returns:

            str: The unique dataset ID.
        """
        return self._lookup_id("dataset",
                               ("width", "height", "tin", "tout"),
                               (width,    height,   tin,   tout),
                               features.dataset, hash_dataset)

    def scenario_id(self, device, kernel, dataset):
        """
        Lookup the ID of a scenario.

        Arguments:

           device (str): Device ID.
           kernel (str): Kernel ID.
           dataset (str): Dataset ID.

        Returns:

           str: The unique scenario ID.
        """
        id = hash_scenario(device, kernel, dataset)
        row = (id,) + (device, kernel, dataset)
        self.execute("INSERT OR IGNORE INTO scenarios VALUES (?,?,?,?)", row)
        return id

    def params_id(self, wg_c, wg_r):
        """
        Lookup the ID of a parameter set.

        Arguments:

           wg_c (int): Workgroup size (columns).
           wg_r (int): Workgroup size (rows).

        Returns:

           str: The unique parameters ID.
        """
        id = hash_scenario(wg_c, wg_r)
        row = (id,) + (device, wg_c, wg_r)
        self.execute("INSERT OR IGNORE INTO params VALUES (?,?,?)", row)
        return id

    def merge(self, rhs):
        """
        Merge the contents of the given database.

        Arguments:

            rhs (Database): Database instance to merge into this.
        """
        self.attach(rhs.path, "rhs")
        self.run("merge_rhs")
        self.detach("rhs")

    def add_runtime(self, scenario, params, runtime):
        """
        Add a new measured experimental runtime.
        """
        self.execute("INSERT INTO runtimes VALUES (?,?,?)",
                     (scenario, params, runtime))

    def _progress_report(self, table_name, i=0, n=1, total=None):
        """
        Intermediate progress updates for long running table jobs.

        Arguments:
            table_name (str): Name of table being operated on.
            i (int, optional): Current row number.
            n (int, optional): Number of rows between printed updates.
            total (int, optional): Total number of rows.
        """
        if total is None:
            io.info("Populating {table} ...".format(table=table_name))
        else:
            if not i % n:
                self.commit()
                io.info("Populating {table} ... {perc:.2f}% ({i} / {total} "
                        "rows).".format(table=table_name,
                                        perc=(i / total) * 100,
                                        i=i, total=total))

    def populate_kernel_names_table(self):
        """
        Derive kernel names from source code.

        Prompts the user interactively in case it needs help.
        """
        kernels = self.kernels
        for i,kernel in enumerate(kernels):
            self._progress_report("kernel_names", i, 10, len(kernels))
            query = self.execute("SELECT id FROM kernel_names WHERE id=?",
                                 (kernel,))

            if not query.fetchone():
                source = self.execute("SELECT source "
                                      "FROM kernel_lookup "
                                      "WHERE id=? LIMIT 1", (kernel,))
                synthetic, name = get_kernel_name_and_type(source)
                self.execute("INSERT INTO kernel_names VALUES (?,?,?)",
                             (kernel, 1 if synthetic else 0, name))

    def populate_runtime_stats_table(self):
        """
        Derive runtime stats from "runtimes" table.
        """
        # Get unique (scenario,param) pairs
        query = self.execute("SELECT scenario,params FROM runtimes "
                             "GROUP BY scenario,params")
        rows = query.fetchall()
        total = len(rows)

        for i,row in enumerate(rows):
            scenario, params = row

            # Gather statistics about runtimes for each scenario,params pair.
            self.execute(self._insert_runtime_stat, (scenario, params))
            self._progress_report("runtime_stats", i, 5, total)
        self.commit()

    def populate_oracle_params_table(self):
        """
        Derive oracle params from "runtime_stats" table.
        """
        # Get unique scenarios.
        scenarios = self.scenarios
        total = len(scenarios)

        for i,scenario in enumerate(scenarios):
            self.execute(self._insert_oracle_param, (scenario, scenario))
            self._progress_report("oracle_params", i, 10, total)

        self.commit()

    def populate_oracle_tables(self):
        """
        Populate the oracle tables.
        """
        self.populate_kernel_names_table()

        self.execute("DELETE FROM runtime_stats")
        self.populate_runtime_stats_table()

        self.execute("DELETE FROM oracle_params")
        self.populate_oracle_params_table()

        io.debug("Compacting database ...")
        self.execute("VACUUM")

        io.info("Done.")

    def lookup_runtimes_count(self, scenario, params):
        """
        Return the number of runtimes for a particular scenario + params.
        """
        query = self.execute("SELECT Count(*) FROM runtimes WHERE "
                             "scenario=? AND params=?", (scenario, params))
        return query.fetchone()[0]

    def lookup_named_kernel(self, name):
        """
        Get the IDs of all kernels with the given name

        Arguments:

            name (str): Kernel name.

        Returns:

            list of str: A list of kernel IDs for the named kernel.
        """
        return [row[0] for row in
                self.execute("SELECT id FROM kernel_names WHERE name=?",
                             (name,))]

    def lookup_named_kernels(self):
        """
        Lookup Kernel IDs by name.

        Returns:

           dict of {str: tuple of str}: Where kernel names are keys,
             and the values are a tuple of kernel IDs with that name.
        """
        return {name: self.lookup_named_kernel(name)
                for name in self.kernel_names}

    def oracle_param_frequencies(self, table="oracle_params",
                                 where=None, normalise=False):
        """
        Return a frequency table of optimal parameter values.

        Arguments:

            table (str, optional): The name of the table to calculate
              the frequencies of.
            normalise (bool, optional): Whether to normalise the
              frequencies, such that the sum of all frequencies is 1.

        Returns:

           list of (str,int) tuples: Where each tuple consists of a
             (params,frequency) pair.
        """
        select = table
        if where: select += " WHERE " + where
        freqs = [row for row in
                 self.execute("SELECT params,Count(*) AS count FROM "
                              "{select} GROUP BY params ORDER BY count ASC"
                              .format(select=select))]

        # Normalise frequencies.
        if normalise:
            total = sum([freq[1] for freq in freqs])
            freqs = [(freq[0], freq[1] / total) for freq in freqs]

        return freqs

    def max_wgsize_frequencies(self, normalise=False):
        """
        Return a frequency table of maximum workgroup sizes.

        Arguments:

            normalise (bool, optional): Whether to normalise the
              frequencies, such that the sum of all frequencies is 1.

        Returns:

           list of (int,int) tuples: Where each tuple consists of a
             (max_wgsize,frequency) pair.
        """
        freqs = [row for row in
                 self.execute("SELECT max_wg_size,Count(*) AS count FROM "
                              "kernels LEFT JOIN scenarios ON "
                              "kernel = kernels.id GROUP BY max_wg_size "
                              "ORDER BY count ASC")]

        # Normalise frequencies.
        if normalise:
            total = sum(freq[1] for freq in freqs)
            freqs = [(freq[0], freq[1] / total) for freq in freqs]

        return freqs

    def oracle_param_space(self, *args, **kwargs):
        """
        Summarise the frequency at which workgroup sizes are optimal.

        Arguments:

            *args, **kwargs: Any additional arguments to be passed to
              oracle_param_frequencies()

        Returns:

            space.ParamSpace: A populated parameter space.
        """
        # Normalise frequencies by default.
        if "normalise" not in kwargs:
            kwargs["normalise"] = True

        freqs = self.oracle_param_frequencies(*args, **kwargs)
        space = ParamSpace(self.wg_c, self.wg_r)

        for wgsize,count in freqs:
            space[wgsize] = count

        return space

    def param_coverage_frequencies(self, **kwargs):
        """
        Return a frequency table of workgroup sizes.

        Arguments:

            **kwargs: Any additional arguments to be passed to param_coverage()

        Returns:

           list of (int,flaot) tuples: Where each tuple consists of a
             (wgsize,frequency) pair.
        """
        return [(param,self.param_coverage(param, **kwargs))
                for param in self.params]

    def param_coverage_space(self, **kwargs):
        """
        Summarise the frequency at workgroup sizes are safe.

        Arguments:

            **kwargs: Any additional arguments to be passed to param_coverage()

        Returns:

            space.ParamSpace: A populated parameter space.
        """
        freqs = self.param_coverage_frequencies(**kwargs)
        space = ParamSpace(self.wg_c, self.wg_r)

        for wgsize,freq in freqs:
            space[wgsize] = freq

        return space

    def param_safeties(self, **kwargs):
        """
        Return a frequency table of workgroup sizes.

        Arguments:

            **kwargs: Any additional arguments to be passed to param_coverage()

        Returns:

           list of (int,bool) tuples: Where each tuple consists of a
             (wgsize,is_safe) pair.
        """
        return [(param, self.param_is_safe(param, **kwargs))
                for param in self.params]

    def param_safe_space(self, **kwargs):
        """
        Summarise the frequency at workgroup sizes are safe.

        Arguments:

            **kwargs: Any additional arguments to be passed to param_coverage()

        Returns:

            space.ParamSpace: A populated parameter space.
        """
        freqs = self.param_safeties(**kwargs)
        space = ParamSpace(self.wg_c, self.wg_r)

        for wgsize,safe in freqs:
            space[wgsize] = 1 if safe else 0

        return space

    def max_wgsize_space(self, *args, **kwargs):
        """
        Summarise the frequency at which workgroup sizes are legal.

        Arguments:

            *args, **kwargs: Any additional arguments to be passed to
              max_wgsize_frequencies()

        Returns:

            space.ParamSpace: A populated parameter space.
        """
        # Normalise frequencies by default.
        if "normalise" not in kwargs:
            kwargs["normalise"] = True

        freqs = self.max_wgsize_frequencies(*args, **kwargs)
        space = ParamSpace(self.wg_c, self.wg_r)

        for maxwgsize,count in freqs:
            for j in range(space.matrix.shape[0]):
                for i in range(space.matrix.shape[1]):
                    wg_r, wg_c = space.r[j], space.c[i]
                    wgsize = wg_r * wg_c
                    if wgsize <= maxwgsize:
                        space.matrix[j][i] += count

        return space

    def num_params_for_scenario(self, scenario):
        """
        Return the number of parameters tested for a given scenario.

        Arguments:

            scenario (str): Scenario ID.

        Returns:

            int: The number of unique parameters sampled for the given
              scenario.
        """
        return self.execute("SELECT Count(*) FROM runtime_stats WHERE "
                            "scenario=?", (scenario,)).fetchone()[0]

    def num_params_for_scenarios(self):
        """
        Return the number of parameters tested for each scenario.

        Returns:

            {str: int} dict: Each key is a scenario, each value is the
              number of parameters sampled for that scenario.
        """
        return {scenario: self.num_params_for_scenario(scenario)
                for scenario in self.scenarios}

    def perf_scenario(self, scenario):
        """
        Return performance of all workgroup sizes relative to oracle.

        Performance relative to the oracle is calculated using mean
        runtime of oracle params / mean runtime of each params.

        Arguments:

            scenario (str): Scenario ID.

        Returns:

            dict of {str: float}: Where each key consists of the
              parameters ID, and each value is the performance of that
              parameter relative to the oracle.
        """
        query = self.execute(self._select_perf_scenario,
                             (scenario, scenario, scenario)).fetchall()
        return {t[0]: t[1] for t in query}

    def perf_param(self, param):
        """
        Return performance of param relative to oracle for all scenarios.

        Performance relative to the oracle is calculated using mean
        runtime of oracle params / mean runtime of each params.

        Arguments:

            param (str): Parameters ID.

        Returns:

            list of (str,float) tuples: Where each tuple consists of
              the parameters ID, and the performance of that parameter
              relative to the oracle.
        """
        query = self.execute(self._select_perf_param,
                             (param,)).fetchall()
        return {t[0]: t[1] for t in query}

    def perf_param_avg(self, param):
        """
        Return the average param performance vs oracle across all scenarios.

        Calculated using the geometric mean of performance relative to
        the oracle of all scenarios.

        Arguments:

            param_id (str): Parameters ID.

        Returns:

            float: Geometric mean of performance relative to oracle.
        """
        return labmath.geomean(self.perf_param(param).values())

    def performance_of_device(self, device):
        """
        Get the performance of all params for all scenarios for a device.

        Arguments:

            device (str): Device ID.

        Returns:

            list of float: Performance of each entry in runtime_Stats
              for that device.
        """
        return lab.flatten([self.perf_scenario(row[0]).values()
                            for row in
                            self.execute("SELECT id FROM scenarios WHERE "
                                         "device=?", (device,))])

    def performance_of_kernels_with_name(self, name):
        """
        Get the performance of all params for all scenarios for a kernel.

        Arguments:

            name (str): Kernel name.

        Returns:

            list of float: Performance of each entry in runtime_stats
              for all kernels with that name.
        """
        kernels = ("(" +
                   ",".join(['"' + id + '"'
                             for id in self.lookup_named_kernel(name)]) +
                   ")")

        return lab.flatten([self.perf_scenario(row[0]).values()
                            for row in
                            self.execute("SELECT id FROM scenarios WHERE "
                                         "kernel IN " + kernels)])

    def performance_of_dataset(self, dataset):
        """
        Get the performance of all params for all scenarios for a dataset.

        Arguments:

            dataset (str): Dataset ID.

        Returns:

            list of float: Performance of each entry in runtime_stats
              for that dataset.
        """
        return lab.flatten([self.perf_scenario(row[0]).values()
                            for row in
                            self.execute("SELECT id FROM scenarios WHERE "
                                         "dataset=?", (dataset,))])

    def oracle_runtime(self, scenario):
        """
        Return the mean runtime of the oracle param for a given scenario.

        Arguments:

            scenario (str): Scenario ID.

        Returns:

            float: Mean runtime of oracle param for scenario.
        """
        return self.execute("SELECT runtime FROM oracle_params WHERE "
                            "scenario=?", (scenario,)).fetchone()[0]

    def runtime(self, scenario, param):
        """
        Return the mean runtime for a given scenario and param.

        Arguments:

            scenario (str): Scenario ID.
            param (str): Parameters ID.

        Returns:

            float: Mean runtime of param for scenario.
        """
        return self.execute("SELECT mean FROM runtime_stats WHERE "
                            "scenario=? AND params=?",
                            (scenario, param)).fetchone()[0]

    def speedup(self, scenario, left, right):
        """
        Return the speedup of left params over right params for scenario.

        Arguments:

            scenario (str): Scenario ID.
            left (str): Parameters ID for left (base) case.
            right (str): Parameters ID for right (comparison) case.

        Returns:

            float: Speedup of left over right.
        """
        left = self.runtime(scenario, left)
        right = self.runtime(scenario, right)
        return left / right

    def perf(self, scenario, param):
        """
        Return the performance of the given param relative to the oracle.

        Arguments:

            scenario (str): Scenario ID.
            param (str): Parameters ID.

        Returns:

            float: Mean runtime for scenario with param, normalised
              against runtime of oracle.
        """
        oracle = self.runtime(scenario, param)
        perf = self.runtime(scenario, param)
        return oracle / perf

    def max_speedup(self, scenario):
        """
        Return the max speedup for a given scenarios.

        The max speedup is the speedup the *best* parameter over the
        *worst* parameter for the scenario.

        Returns:

            float: Max speedup for scenario.
        """
        best = self.execute("SELECT runtime\n"
                            "FROM oracle_params\n"
                            "WHERE scenario=?",
                            (scenario,)).fetchone()[0]
        worst = self.execute("SELECT\n"
                             "    mean AS runtime\n"
                             "FROM runtime_stats\n"
                             "WHERE\n"
                             "    scenario=? AND\n"
                             "    mean=(\n"
                             "        SELECT MAX(mean)\n"
                             "        FROM runtime_stats\n"
                             "        WHERE scenario=?\n"
                             ")",
                             (scenario, scenario)).fetchone()[0]
        return best / worst

    def max_speedups(self):
        """
        Return the max speedups for all scenarios.

        The max speedup is the speedup the *best* parameter over the
        *worst* parameter for a given scenario.

        Returns:

            dict of {str: float}: Where the keys are scenario IDs, and
              the values are max speedup of that scenario.
        """
        return {scenario: self.max_speedup(scenario)
                for scenario in self.scenarios}

    def min_max_runtime(self, scenario, params):
        """
        Return the ratio of min and max runtimes to the mean.

        Arguments:

            scenario (str): Scenario ID.
            parms (str): Parameters ID.

        Returns:

            (float, float): The minimum and maximum runtimes,
              normalised against the mean.
        """
        min_t, mean_t, max_t = self.execute("SELECT min,mean,max "
                                            "FROM runtime_stats "
                                            "WHERE scenario=? AND params=?",
                                            (scenario, params)).fetchone()
        return min_t / mean_t, max_t / mean_t

    def min_max_runtimes(self):
        """
        Return the min/max runtimes for all scenarios and params.

        Returns:

            list of (str,str,float,float): Where the first element is
              the scenario ID, the second is the param ID, and the
              third and fourth are the normalised min max runtimes.
        """
        return [row for row in
                self.execute("SELECT\n"
                             "    scenario,\n"
                             "    params,\n"
                             "    (min / mean),\n"
                             "    (max / mean)\n"
                             "FROM runtime_stats")]

    def num_samples(self):
        """
        Return the number of samples for all scenarios and params.

        Returns:

            list of (str,str,int): Where the first item is the
              scenario ID, the second the params ID, the third the
              number of samples for that (scenario, params) pair.
        """
        return [row for row in
                self.execute("SELECT\n"
                             "    scenario,\n"
                             "    params,\n"
                             "    num_samples\n"
                             "FROM runtime_stats\n"
                             "ORDER BY num_samples DESC")]

    def params_summary(self):
        """
        Return a summary of parameters.

        Returns:

            list of (str,float,float) tuples: Where each tuple is of
              the format (param_id,perforance,coverage).
        """
        return sorted([(param,
                        self.perf_param_avg(param),
                        self.param_coverage(param)) for param in self.params],
                      key=lambda t: t[1], reverse=True)

    def param_coverage(self, param_id, where=None):
        """
        Returns the ratio of values for a params across scenarios.

        Arguments:

            param (str): Parameters ID.

        Returns:

            float: Number of scenarios with recorded values for parm /
              total number of scenarios.
        """
        # Get the total number of scenarios.
        select = "SELECT Count(*) FROM (SELECT id as scenario from scenarios)"
        if where:
            select += " WHERE " + where
        num_scenarios = self.execute(select).fetchone()[0]

        # Get the ratio of runtimes to total where params = param_id.
        select = ("SELECT (CAST(Count(*) as REAL) / CAST(? AS REAL)) "
                  "FROM runtime_stats WHERE params=?")
        if where:
            select += " AND " + where
        return self.execute(select, (num_scenarios, param_id)).fetchone()[0]

    def param_is_safe(self, param_id, **kwargs):
        """
        Returns whether a parameter is safe.

        A parameter is safe if, for all scenarios, there is recorded
        runtimes. This implies that the parameter is valid for all
        possible cases.

        Arguments:

            param (str): Parameters ID.
            **kwargs: Any additional arguments to be passed to param_coverage()

        Returns:

            bool: True if parameter is safe, else false.
        """
        return self.param_coverage(param_id, **kwargs) == 1

    def zero_r(self):
        """
        Return the ZeroR parameter, and its performance.

        The ZeroR parameter is the parameter which is most often
        optimal.

        Returns:

           (str, float): Where str if the parameters ID of the ZeroR,
             and float is the average performance relative to the
             oracle.
        """
        zeror = self.execute("SELECT params,Count(params) AS count "
                             "FROM oracle_params GROUP BY params "
                             "ORDER BY count DESC LIMIT 1").fetchone()[0]

        return zeror, self.perf_param_avg(zeror)

    def one_r(self):
        """
        Return the OneR parameter, and its performance.

        The OneR parameter is the parameter which provides the best
        average case performance across all scenarios. Average
        performance is calculated using the geometric mean of
        performance relative to the oracle.

        Returns:

           (str, float): Where str if the parameters ID of the OneR,
             and float is the average performance relative to the
             oracle.
        """
        # Get average performance of each param.
        avgs = [(param, self.perf_param_avg(param)) for param in self.params]
        # Return the "best" param
        return max(avgs, key=lambda x: x[1])

    def dump_csvs(self, path="."):
        """
        Dump CSV files.

        Arguments:

            path (str, optional): Directory to dump CSV files
              into. Defaults to working directory.
        """
        fs.mkdir(path)
        fs.cd(path)
        self.runscript("dump_csvs")
        fs.cdpop()


def create_test_db(dst, src, num_runtimes=100000):
    """
    Create a reduced-size database for testing.

    A copy of the source database is made, but "num_runtimes" are
    selected randomly. This is to allow faster testing on smaller
    databases.

    Arguments:

        dst (path): The path to the destination database.
        src (Database): The source database.
        num_runtimes (int, optional): The maximum number of runtimes
          to keep.

    Returns:

        Database: The reduced test database.
    """
    io.info("Creating test database of {n} runtimes"
            .format(n=num_runtimes))

    fs.cp(src.path, dst)
    test = Database(dst)

    # Copy old runtimes table.
    test.copy_table("runtimes", "runtimes_tmp")
    test.drop_table("runtimes")

    # Create new runtimes table.
    test.create_table_from("runtimes", "runtimes_tmp")
    cmd = ("INSERT INTO runtimes SELECT * FROM runtimes_tmp "
           "ORDER BY RANDOM() LIMIT {n}".format(n=num_runtimes))
    test.execute(cmd)
    test.drop_table("runtimes_tmp")

    # Remove unused scenarios.
    test.execute("DELETE FROM scenarios WHERE id NOT IN "
                 "(SELECT DISTINCT scenario from runtimes)")

    # Remove unused kernels.
    test.execute("DELETE FROM kernels WHERE id NOT IN "
                 "(SELECT DISTINCT kernel from scenarios)")
    # Remove unused devices.
    test.execute("DELETE FROM devices WHERE id NOT IN "
                 "(SELECT DISTINCT device from scenarios)")
    # Remove unused datasets.
    test.execute("DELETE FROM datasets WHERE id NOT IN "
                 "(SELECT DISTINCT dataset from scenarios)")

    test.commit()

    # Shrink database to reclaim lost space.
    test.execute("VACUUM")

    return test
