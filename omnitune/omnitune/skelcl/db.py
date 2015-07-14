from __future__ import division
from __future__ import print_function

import csv
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
from . import hash_classifier
from . import hash_err_fn
from . import hash_ml_dataset
from . import hash_ml_job

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


def _merge_min(lhs, rhs):
    return min(lhs, rhs)

def _merge_mean(lhs, nl, rhs, nr):
    n = nl + nr
    return (lhs * nl + rhs * nr) / n

def _merge_max(lhs, rhs):
    return max(lhs, rhs)


class ItemAggregator(object):
    def __init__(self):
        self.items = []

    def step(self, value):
        self.items.append(value)


class GeomeanAggregate(ItemAggregator):
    def finalize(self):
        return labmath.geomean(self.items)


class ConfErrorAggregate(ItemAggregator):
    def step(self, value, conf):
        super(ConfErrorAggregate, self).step(value)
        self.conf = conf

    def finalize(self):
        return labmath.confinterval(self.items, self.conf, error_only=True)


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
        self._select_perf_param_legal = sql_command("select_perf_param_legal")
        self._select_ratio_max_wgsize = sql_command("select_ratio_max_wgsize")
        self._best_average_classification_performance = sql_command(
            "best_average_classification_performance")
        self._best_average_classification_speedup = sql_command(
            "best_average_classification_speedup")
        self._best_average_synthetic_real_classification_performance = sql_command(
            "best_average_synthetic_real_classification_performance")
        self._best_average_synthetic_real_classification_speedup = sql_command(
            "best_average_synthetic_real_classification_speedup")

        self.connection.create_aggregate("geomean", 1, GeomeanAggregate)
        self.connection.create_aggregate("conferror", 2, ConfErrorAggregate)
        self.connection.create_function("merge_min", 2, _merge_min)
        self.connection.create_function("merge_mean", 4, _merge_mean)
        self.connection.create_function("merge_max", 2, _merge_max)

        if self.isempty():
            self.create_tables()

        # Get the database version.
        try:
            # Look up the version in the table.
            query = self.execute("SELECT version from version")
            self.version = query.fetchone()[0]
        except Exception:
            # Base case: This is pre-versioning.
            self.version = 0

    def status_report(self):
        io.info("Database status:")
        io.info("    Number of runtimes:   " + str(self.num_rows("runtimes")))
        io.info("    Number of params:     " + str(self.num_rows("params")))
        io.info("    Number of scenarios:  " + str(self.num_rows("scenarios")))
        io.info("    Number of kernels:    " + str(self.num_rows("kernels")))
        io.info("    Number of devices:    " + str(self.num_rows("devices")))

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
    def real_kernels(self):
        return [row[0] for row in
                self.execute("SELECT id FROM kernel_names WHERE synthetic=1")]

    @property
    def synthetic_kernels(self):
        return [row[0] for row in
                self.execute("SELECT id FROM kernel_names WHERE synthetic=0")]

    @property
    def num_scenarios(self):
        return self.execute("SELECT Count(*) from scenarios").fetchone()[0]

    @property
    def scenarios(self):
        return [row[0] for row in
                self.execute("SELECT id FROM scenarios")]

    @property
    def real_scenarios(self):
        return [row[0] for row in
                self.execute("SELECT id FROM scenarios WHERE kernel IN"
                             "(SELECT id FROM kernel_names WHERE synthetic=1)")]

    @property
    def synthetic_scenarios(self):
        return [row[0] for row in
                self.execute("SELECT id FROM scenarios WHERE kernel IN"
                             "(SELECT id FROM kernel_names WHERE synthetic=0)")]

    @property
    def scenario_properties(self, where=None):
        """
        Return a list of scenario descriptions.

        Returns:

            list of tuples: Elements in tuple:

                scenario,device,kernel,north,south,east,west,width,height,tout
        """
        query = ("SELECT\n"
                 "    scenarios.id,\n"
                 "    devices.name,\n"
                 "    kernel_names.name,\n"
                 "    kernels.north,\n"
                 "    kernels.south,\n"
                 "    kernels.east,\n"
                 "    kernels.west,\n"
                 "    datasets.width,\n"
                 "    datasets.height,\n"
                 "    datasets.tout\n"
                 "FROM scenarios\n"
                 "LEFT JOIN devices\n"
                 "    ON scenarios.device=devices.id\n"
                 "LEFT JOIN kernel_names\n"
                 "    ON scenarios.kernel=kernel_names.id\n"
                 "LEFT JOIN kernels\n"
                 "    ON scenarios.kernel=kernels.id\n"
                 "LEFT JOIN datasets\n"
                 "    ON scenarios.dataset=datasets.id\n"
                 "ORDER BY scenarios.device ASC")
        if where is not None:
            query += " WHERE " + where
        return self.execute(query).fetchall()

    @property
    def mean_samples(self):
        return self.execute("SELECT AVG(num_samples) FROM "
                            "runtime_stats").fetchone()[0]

    @property
    def scenario_params(self):
        return [row for row in
                self.execute("SELECT scenario,params FROM "
                             "runtime_stats GROUP BY scenario,params")]

    @property
    def oracle_params(self):
        return [(row[0], int(row[1])) for row in
                self.execute("SELECT\n"
                             "    params,\n"
                             "    Count(params) AS count\n"
                             "FROM oracle_params\n"
                             "GROUP BY params\n"
                             "ORDER BY count DESC")]

    @property
    def classifiers(self):
        return [row[0] for row in self.execute("SELECT id FROM classifiers")]

    @property
    def classification_classifiers(self):
        return [row[0] for row in
                self.execute("SELECT DISTINCT classifier FROM "
                             "classification_results")]

    @property
    def regression_classifiers(self):
        return [row[0] for row in
                self.execute("SELECT DISTINCT classifier FROM "
                             "runtime_regression_results")]

    @property
    def err_fns(self):
        return [row[0] for row in
                self.execute("SELECT DISTINCT err_fn FROM "
                             "classification_results")]

    @property
    def classifier_err_fns(self):
        return [row for row in
                self.execute("SELECT classifier,err_fn FROM "
                             "classification_results "
                             "GROUP BY classifier,err_fn")]

    @property
    def best_average_classification_performance(self):
        return self.execute(
            self._best_average_classification_performance
        ).fetchone()[0]

    @property
    def best_average_classification_speedup(self):
        return self.execute(
            self._best_average_classification_speedup
        ).fetchone()[0]

    @property
    def best_average_synthetic_real_classification_performance(self):
        return self.execute(
            self._best_average_synthetic_real_classification_performance
        ).fetchone()[0]

    @property
    def best_average_synthetic_real_classification_speedup(self):
        return self.execute(
            self._best_average_synthetic_real_classification_speedup
        ).fetchone()[0]

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
        query = self.execute("SELECT id\n"
                             "FROM " + lookup_table + "\n"
                             "WHERE " + where(*lookup_columns),
                             lookup_vals).fetchone()

        # If there's an entry in the lookup table, return.
        if query:
            return query[0]

        # If there's no entry in lookup table, perform feature extraction.
        features = feature_fn(*lookup_vals)
        id = id_fn(*features)

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
        return self._id("dataset",
                        ("width", "height", "tin", "tout"),
                        ( width,   height,   tin,   tout),
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
        id = hash_params(wg_c, wg_r)
        row = (id,) + (wg_c, wg_r)
        self.execute("INSERT OR IGNORE INTO params VALUES (?,?,?)", row)
        return id

    def _merge_rhs(self, rhs):
        io.info("Merging", rhs.path)
        self.attach(rhs.path, "rhs")

        # Import runtimes,features,datasets tables.
        # Populate rhs runtime_stats table.
        self.run("merge_rhs")

        rows = [row for row in
                self.execute("SELECT * FROM rhs.runtime_stats")]
        total = len(rows)

        # Insert or merge the contents of the rhs.runtime_stats table.
        # TODO: This could probably be implemented using an SQL
        # query, if I had the time and the know-how.
        for i,row in enumerate(rows):
            self._progress_report("runtime_stats", i, 10, total)
            scenario, params, rhs_count, rhs_min, rhs_mean, rhs_max = row
            lhs = self.execute("SELECT num_samples,min,mean,max\n"
                               "FROM runtime_stats\n"
                               "WHERE scenario=? AND params=?",
                               (scenario, params)).fetchone()
            if lhs:
                # Prior value, so update the existing value.
                lhs_count, lhs_min, lhs_mean, lhs_max = lhs

                new_count = lhs_count + rhs_count
                new_min = min(lhs_min, rhs_min)
                new_mean = ((lhs_mean * lhs_count + rhs_mean * rhs_count)
                            / new_count)
                new_max = max(lhs_max, rhs_max)

                self.execute("UPDATE runtime_stats\n"
                             "SET num_samples=?,min=?,mean=?,max=?\n"
                             "WHERE scenario=? AND params=?",
                             (new_count, new_min, new_mean, new_max,
                              scenario, params))
            else:
                # No prior value, so just add a new row.
                self.execute("INSERT INTO runtime_stats\n"
                             "VALUES (?,?,?,?,?,?)",
                             (scenario, params, rhs_count,
                              rhs_min, rhs_mean, rhs_max))

        self.commit()
        self.detach("rhs")

    def merge(self, dbs):
        """
        Merge the contents of the given databases.

        Arguments:

            dbs (list of Database objects): Database instances to
              merge into this.
        """
        for db in dbs:
            self._merge_rhs(db)

        io.info("Updating oracle tables ...")
        self.execute("DELETE FROM oracle_params")
        self.populate_oracle_params_table()

        io.debug("Compacting database ...")
        self.execute("VACUUM")

        io.info("Done.")

    def add_runtime(self, scenario, params, runtime):
        """
        Add a new measured experimental runtime.
        """
        self.execute("INSERT INTO runtimes VALUES (?,?,?)",
                     (scenario, params, runtime))

    def add_classification_result(self, job, classifier, err_fn, dataset,
                                  scenario, actual, predicted, baseline,
                                  correct, invalid, performance, speedup):
        """
        Add result of using a classifier to predict optimal workgroup size.

        Arguments:

            job (str): ML job name.
            classifier (ml.Classifier): Classifier.
            err_fn (function partial): Error handler function.
            dataset (WekaInstances): Classifier training data.
            scenario (str): Scenario ID.
            actual (str): Params ID of oracle.
            predicted (str): Params ID of classifier prediction.
            baseline (str): Params ID of baseline for performance baseline.
            correct (int): 1 if prediction is correct.
            invalid (int): 1 if *first* prediction was legal.
            performance (float): Performance relative to oracle of prediction.
            speedup (float): Speedup over performance baseline.
        """
        job_id = self.ml_job_id(job)
        classifier_id = self.classifier_id(classifier)
        err_fn_id = self.err_fn_id(err_fn)
        dataset_id = self.ml_dataset_id(dataset)

        self.execute("INSERT INTO classification_results VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (job_id, classifier_id, err_fn_id, dataset_id,
                      scenario, actual, predicted, baseline, correct,
                      invalid, performance, speedup))

    def add_runtime_regression_result(self, job, classifier, dataset, scenario,
                                      params, actual, predicted, norm_predicted,
                                      norm_error):
        """
        Add result of using a regressor to predict stencil runtime.

        Arguments:

            job (str): ML job name.
            classifier (ml.Classifier): Classifier.
            dataset (WekaInstances): Classifier training data.
            scenario (str): Scenario ID.
            params (str): Parameters ID.
            actual (float): Actual measured mean runtime.
            predicted (float): Predicted mean runtime.
            norm_predicted (float): Predicted runtime, normalised against actual.
            norm_error (float): abs(norm_predicted - 1)
        """
        job_id = self.ml_job_id(job)
        classifier_id = self.classifier_id(classifier)
        dataset_id = self.ml_dataset_id(dataset)

        self.execute("INSERT INTO runtime_regression_results VALUES "
                     "(?,?,?,?,?,?,?,?,?)",
                     (job_id, classifier_id, dataset_id, scenario, params,
                      actual, predicted, norm_predicted, norm_error))

    def add_runtime_classification_result(self, job, classifier, scenario,
                                          actual, predicted, baseline,
                                          correct, performance, speedup):
        """
        Add result of using a runtime regressor to predict optimal
        workgroup size.

        Arguments:

            job (str): ML job name.
            classifier (ml.Classifier): Classifier.
            scenario (str): Scenario ID.
            actual (str): Params ID of oracle.
            predicted (str): Params ID of classifier prediction.
            baseline (str): Params ID of baseline for performance baseline.
            correct (int): 1 if prediction is correct.
            performance (float): Performance relative to oracle of prediction.
            speedup (float): Speedup over performance baseline.
        """
        job_id = self.ml_job_id(job)
        classifier_id = self.classifier_id(classifier)

        self.execute("INSERT INTO runtime_classification_results VALUES "
                     "(?,?,?,?,?,?,?,?,?)",
                     (job_id, classifier_id, scenario, actual, predicted,
                      baseline, correct, performance, speedup))

    def add_speedup_regression_result(self, job, classifier, dataset, scenario,
                                      params, actual, predicted, norm_predicted,
                                      norm_error):
        """
        Add result of using a regressor to predict relative performance stencil.

        Arguments:

            job (str): ML job name.
            classifier (ml.Classifier): Classifier.
            dataset (WekaInstances): Classifier training data.
            scenario (str): Scenario ID.
            params (str): Parameters ID.
            actual (float): Actual measured speedup.
            predicted (float): Predicted mean speedup.
            norm_predicted (float): Predicted speedup, normalised against actual.
            norm_error (float): abs(norm_predicted - 1)
        """
        job_id = self.ml_job_id(job)
        classifier_id = self.classifier_id(classifier)
        dataset_id = self.ml_dataset_id(dataset)

        self.execute("INSERT INTO speedup_regression_results VALUES "
                     "(?,?,?,?,?,?,?,?,?)",
                     (job_id, classifier_id, dataset_id, scenario, params,
                      actual, predicted, norm_predicted, norm_error))

    def add_speedup_classification_result(self, job, classifier,
                                          scenario, actual, predicted, baseline,
                                          correct, performance, speedup):
        """
        Add result of using a speedup regressor to predict optimal
        workgroup size.

        Arguments:

            job (str): ML job name.
            classifier (ml.Classifier): Classifier.
            scenario (str): Scenario ID.
            actual (str): Params ID of oracle.
            predicted (str): Params ID of classifier prediction.
            baseline (str): Params ID of baseline for performance baseline.
            correct (int): 1 if prediction is correct.
            performance (float): Performance relative to oracle of prediction.
            speedup (float): Speedup over performance baseline.
        """
        job_id = self.ml_job_id(job)
        classifier_id = self.classifier_id(classifier)

        self.execute("INSERT INTO speedup_classification_results VALUES "
                     "(?,?,?,?,?,?,?,?,?)",
                     (job_id, classifier_id, scenario, actual, predicted,
                      baseline, correct, performance, speedup))

    def ml_job_id(self, name):
        """
        Return the ml job id
        """
        id = hash_ml_job(name)
        self.execute("INSERT OR IGNORE INTO ml_jobs VALUES (?)", (id,))
        return id

    def classifier_id(self, classifier):
        """
        Return the classifier ID.
        """
        # FIXME: We should be using hash_classifier() !!!
        classifier_id = str(classifier)
        classname = classifier.classname
        options = " ".join(classifier.options)
        self.execute("INSERT OR IGNORE INTO classifiers VALUES (?,?,?)",
                     (classifier_id, classname, options))
        return classifier_id

    def err_fn_id(self, err_fn):
        id = hash_err_fn(err_fn)
        self.execute("INSERT OR IGNORE INTO err_fns VALUES (?)", (id,))
        return id

    def ml_dataset_id(self, dataset):
        id = hash_ml_dataset(dataset)
        data = str(dataset)
        self.execute("INSERT OR IGNORE INTO ml_datasets VALUES (?,?)",
                     (id, data))
        return id

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
                                      "WHERE id=? LIMIT 1", (kernel,)).fetchone()[0]
                synthetic, name = get_kernel_name_and_type(source)
                self.execute("INSERT INTO kernel_names VALUES (?,?,?)",
                             (kernel, 1 if synthetic else 0, name))
                self.commit()

    def populate_runtime_stats_table(self):
        """
        Derive runtime stats from "runtimes" table.
        """
        self.run("populate_runtime_stats")
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

           dict of {str: int} pairs: Where the keys are parameters and
             the values are frequencies.
        """
        select = [table]
        if where:
            select.append("WHERE")
            select.append(where)
        freqs = {
            row[0]: row[1] for row in
            self.execute("SELECT params,Count(*) AS count FROM "
                         "{select} GROUP BY params ORDER BY count ASC"
                         .format(select=" ".join(select)))
        }

        # Normalise frequencies.
        if normalise:
            total = sum(freqs.values())
            for freq in freqs:
                freqs[freq] /= total

        return freqs

    def rand_wgsize(self, max_wgsize=0):
        """
        Fetch a random wgsize pair.

        Arguments:

            max_wgsize (int, optional): If greater than 0, return a
              workgroup size <= max_wgsize.

        Returns:

            (int, int): wg_c and wg_r parameter values, in that order.
        """
        query = ["SELECT wg_c,wg_r FROM params"]

        # Limit maximum wgsize.
        if max_wgsize > 0:
            query.append("WHERE wg_c * wg_r <=")
            query.append(str(max_wgsize))

        query.append("ORDER BY RANDOM() LIMIT 1")

        return self.execute(" ".join(query)).fetchone()


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
        return ParamSpace.from_dict(freqs, wg_c=self.wg_c, wg_r=self.wg_r)


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

    def scenarios_for_device(self, device):
        """
        Return the scenario IDs for a given device.

        Arguments:

            device (str): Device ID.

        Returns:

            list of str: List of scenario IDs.
        """
        return [
            row[0] for row in
            self.execute("SELECT id FROM scenarios WHERE device=?",
                         (device,))
        ]

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
        return {scenario: self.perf(scenario, param)
                for scenario in self.scenarios}

    def perf_param_legal(self, param):
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
        query = self.execute(self._select_perf_param_legal,
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

    def perf_param_avg_legal(self, param):
        """
        Param performance across all scenarios for which param was legal.

        Calculated using the geometric mean of performance relative to
        the oracle for the subset of scenarios in which the param was
        legal.

        Arguments:

            param_id (str): Parameters ID.

        Returns:

            float: Geometric mean of performance relative to oracle.
        """
        return labmath.geomean(self.perf_param_legal(param).values())

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

    def oracle_param(self, scenario):
        # TODO: Document!
        """
        """
        return self.execute("SELECT params FROM oracle_params WHERE scenario=?",
                            (scenario,)).fetchone()[0]

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

    def runtime(self, scenario, param, default=None):
        """
        Return the mean runtime for a given scenario and param.

        Arguments:

            scenario (str): Scenario ID.
            param (str): Parameters ID.

        Returns:

            float: Mean runtime of param for scenario.
        """
        try:
            return self.execute("SELECT mean\n"
                                "FROM runtime_stats\n"
                                "WHERE scenario=? AND params=?",
                                (scenario, param)).fetchone()[0]
        except TypeError as err:
            if default is not None:
                return default
            else:
                raise db.Error("No runtime information for {scenario} {params}"
                               .format(scenario=scenario, params=param))

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
        oracle = self.runtime(scenario, self.oracle_param(scenario))
        perf = self.runtime(scenario, param, 0)

        if perf > 0:
            return oracle / perf
        else:
            return 0

    def ratio_max_wgsize(self, scenario, param):
        """
        Return the ratio of the given param size to the max legal.
        """
        return self.execute(sql_command("select_ratio_max_wgsize"),
                            (param, scenario)).fetchone()[0]

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
                             "    )",
                             (scenario, scenario)).fetchone()[0]
        return worst / best

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
        return self.execute("SELECT\n"
                            "    (min / mean),\n"
                            "    (max / mean),\n"
                            "FROM runtime_stats "
                            "WHERE scenario=? AND params=?",
                            (scenario, params)).fetchone()

    def min_max_runtimes(self, where=None):
        """
        Return the min/max runtimes for all scenarios and params.

        Returns:

            list of (str,str,float,float): Where the first element is
              the scenario ID, the second is the param ID, and the
              third and fourth are the normalised min max runtimes.
        """
        query = ("SELECT\n"
                 "    scenario,\n"
                 "    params,\n"
                 "    (min / mean),\n"
                 "    (max / mean)\n"
                 "FROM runtime_stats")
        if where is not None:
            query += " WHERE " + where
        return [row for row in self.execute(query)]

    def num_samples(self, where=None):
        """
        Return the number of samples for all scenarios and params.

        Returns:

            list of (str,str,int): Where the first item is the
              scenario ID, the second the params ID, the third the
              number of samples for that (scenario, params) pair.
        """
        query = ("SELECT\n"
                 "    scenario,\n"
                 "    params,\n"
                 "    num_samples\n"
                 "FROM runtime_stats\n"
                 "ORDER BY num_samples DESC")
        if where is not None:
            query += " WHERE " + where
        return [row for row in self.execute(query)]

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

           (str, float, float): Where first element is the parameters
             ID of the OneR, the second element is the ratio of times
             this parameter was optimal, and the third parameter is
             the average performance of the parameter relative to the
             oracle.
        """
        oracles = self.oracle_params
        num_oracles = sum(x[1] for x in oracles)
        best_wgsize, best_count = oracles[0]
        best_perf = self.perf_param_avg(best_wgsize)
        return (best_wgsize, best_count / num_oracles, best_perf)

    def one_r(self):
        """
        Return the OneR parameter, and its performance.

        The OneR parameter is the parameter which provides the best
        average case performance across all scenarios. Average
        performance is calculated using the geometric mean of
        performance relative to the oracle.

        Returns:

           (str, float, float): Where first element is the parameters
             ID of the OneR, the second element is the ratio of times
             this parameter was optimal, and the third parameter is
             the average performance of the parameter relative to the
             oracle.
        """
        def _num_times_optimal(param):
            oracles = self.oracle_params
            for row in oracles:
                wgsize, count = row
                if wgsize == param:
                    return count
            return 0

        # Get average performance of each param.
        avgs = [(param, self.perf_param_avg(param)) for param in self.params]
        best_wgsize, best_perf = max(avgs, key=lambda x: x[1])
        num_oracles = sum([x[1] for x in self.oracle_params])
        best_count = _num_times_optimal(best_wgsize)

        # Return the "best" param
        return best_wgsize, best_count / num_oracles, best_perf

    def oracle_speedups(self):
        """
        Return speedups over the OneR for all parameters.

        Returns:

            dict of {str: float}: Where the keys are scenario IDs, and
              the keys are speedups of the oracle params over the
              OneR.
        """
        one_r = self.one_r()[0]
        return {
            scenario: self.speedup(scenario, one_r, self.oracle_param(scenario))
            for scenario in self.scenarios
        }

    def runtime_predictions(self, scenario, classifier, job):
        clsasifier_id = self.classifier_id(classifier)
        return [
            row for row in
            self.execute(
                "SELECT params,predicted "
                "FROM runtime_regression_results "
                "WHERE job=? AND classifier=? AND scenario=?",
                (job, classifier_id, scenario)
            )
        ]

    def speedup_predictions(self, scenario, classifier, job):
        clsasifier_id = self.classifier_id(classifier)
        return [
            row for row in
            self.execute(
                "SELECT params,predicted "
                "FROM speedup_regression_results "
                "WHERE job=? AND classifier=? AND scenario=?",
                (job, classifier_id, scenario)
            )
        ]

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

        # Derive speedup CSV from runtime stats.
        one_r = self.one_r()[0]
        with open("speedup_stats.csv", "wb") as outfile:
            writer = csv.writer(outfile)
            with open("runtime_stats.csv", "rb") as infile:
                reader = csv.reader(infile)
                header = True
                for row in reader:
                    if header:
                        writer.writerow(row)
                        header = False
                        continue
                    scenario = row[0]
                    params = row[-2]
                    row[-1] = self.speedup(scenario, one_r, params)
                    writer.writerow(row)

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
