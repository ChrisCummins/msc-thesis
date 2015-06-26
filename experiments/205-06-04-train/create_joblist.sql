--
-- Create a database 2xTahiti.db and move any 2xTahiti results into
-- it.
--

ATTACH "joblist.db" as rhs;

CREATE TABLE rhs.jobs (
    scenario                        TEXT,
    kernel                          TEXT,
    north                           INTEGER,
    south                           INTEGER,
    east                            INTEGER,
    west                            INTEGER,
    width                           INTEGER,
    height                          INTEGER,
    device                          TEXT,
    params                          TEXT,
    PRIMARY KEY (scenario,params)
);

INSERT INTO rhs.jobs
SELECT
    scenarios.id,
    kernel_names.name,
    kernels.north,
    kernels.south,
    kernels.east,
    kernels.west,
    datasets.width,
    datasets.height,
    scenarios.device,
    params.id
FROM scenarios
LEFT JOIN params
LEFT JOIN runtime_stats
    ON scenarios.id=runtime_stats.scenario AND params.id=runtime_stats.params
LEFT JOIN kernels
    ON scenarios.kernel=kernels.id
LEFT JOIN kernel_names
    ON kernels.id=kernel_names.id
LEFT JOIN datasets
    ON scenarios.dataset=datasets.id
WHERE
    (params.wg_c * params.wg_r) < kernels.max_wg_size
    AND runtime_stats.num_samples IS NULL;

DETACH rhs;
