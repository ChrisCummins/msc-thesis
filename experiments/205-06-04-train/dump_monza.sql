--
-- Create a database 2xTahiti.db and move any 2xTahiti results into
-- it.
--

ATTACH "2xTahiti.db" as rhs;

-- Version table
CREATE TABLE rhs.version (
    version                         INTEGER,
    PRIMARY KEY (version)
);
INSERT INTO rhs.version VALUES (3);

-- Devices table
CREATE TABLE rhs.devices (
    id                              TEXT,
    name                            TEXT,
    count                           INTEGER,
    address_bits                    INTEGER,
    double_fp_config                TEXT,
    endian_little                   TEXT,
    execution_capabilities          TEXT,
    extensions                      TEXT,
    global_mem_cache_size           INTEGER,
    global_mem_cache_type           TEXT,
    global_mem_cacheline_size       INTEGER,
    global_mem_size                 INTEGER,
    host_unified_memory             TEXT,
    image2d_max_height              INTEGER,
    image2d_max_width               INTEGER,
    image3d_max_depth               INTEGER,
    image3d_max_height              INTEGER,
    image3d_max_width               INTEGER,
    image_support                   TEXT,
    local_mem_size                  INTEGER,
    local_mem_type                  TEXT,
    max_clock_frequency             INTEGER,
    max_compute_units               INTEGER,
    max_constant_args               INTEGER,
    max_constant_buffer_size        INTEGER,
    max_mem_alloc_size              INTEGER,
    max_parameter_size              INTEGER,
    max_read_image_args             INTEGER,
    max_samplers                    INTEGER,
    max_work_group_size             INTEGER,
    max_work_item_dimensions        INTEGER,
    max_work_item_sizes_0           INTEGER,
    max_work_item_sizes_1           INTEGER,
    max_work_item_sizes_2           INTEGER,
    max_write_image_args            INTEGER,
    mem_base_addr_align             INTEGER,
    min_data_type_align_size        INTEGER,
    native_vector_width_char        INTEGER,
    native_vector_width_double      INTEGER,
    native_vector_width_float       INTEGER,
    native_vector_width_half        INTEGER,
    native_vector_width_int         INTEGER,
    native_vector_width_long        INTEGER,
    native_vector_width_short       INTEGER,
    preferred_vector_width_char     INTEGER,
    preferred_vector_width_double   INTEGER,
    preferred_vector_width_float    INTEGER,
    preferred_vector_width_half     INTEGER,
    preferred_vector_width_int      INTEGER,
    preferred_vector_width_long     INTEGER,
    preferred_vector_width_short    INTEGER,
    queue_properties                TEXT,
    single_fp_config                TEXT,
    type                            TEXT,
    vendor                          TEXT,
    vendor_id                       TEXT,
    version                         TEXT,
    PRIMARY KEY (id)
);

-- Kernels table
CREATE TABLE rhs.kernels (
    id                              TEXT,
    north                           INTEGER,
    south                           INTEGER,
    east                            INTEGER,
    west                            INTEGER,
    max_wg_size                     INTEGER,
    instruction_count               INTEGER,
    ratio_AShr_insts                REAL,
    ratio_Add_insts                 REAL,
    ratio_Alloca_insts              REAL,
    ratio_And_insts                 REAL,
    ratio_Br_insts                  REAL,
    ratio_Call_insts                REAL,
    ratio_FAdd_insts                REAL,
    ratio_FCmp_insts                REAL,
    ratio_FDiv_insts                REAL,
    ratio_FMul_insts                REAL,
    ratio_FPExt_insts               REAL,
    ratio_FPToSI_insts              REAL,
    ratio_FSub_insts                REAL,
    ratio_GetElementPtr_insts       REAL,
    ratio_ICmp_insts                REAL,
    ratio_InsertValue_insts         REAL,
    ratio_Load_insts                REAL,
    ratio_Mul_insts                 REAL,
    ratio_Or_insts                  REAL,
    ratio_PHI_insts                 REAL,
    ratio_Ret_insts                 REAL,
    ratio_SDiv_insts                REAL,
    ratio_SExt_insts                REAL,
    ratio_SIToFP_insts              REAL,
    ratio_SRem_insts                REAL,
    ratio_Select_insts              REAL,
    ratio_Shl_insts                 REAL,
    ratio_Store_insts               REAL,
    ratio_Sub_insts                 REAL,
    ratio_Trunc_insts               REAL,
    ratio_UDiv_insts                REAL,
    ratio_Xor_insts                 REAL,
    ratio_ZExt_insts                REAL,
    ratio_basic_blocks              REAL,
    ratio_memory_instructions       REAL,
    ratio_non_external_functions    REAL,
    PRIMARY KEY (id)
);

-- Scenarios table
CREATE TABLE rhs.scenarios (
    id                              TEXT,
    device                          TEXT,
    kernel                          TEXT,
    dataset                         TEXT,
    PRIMARY KEY (id)
);

-- Runtimes table
CREATE TABLE rhs.runtimes (
    scenario                        TEXT,
    params                          TEXT,
    runtime                         REAL
);

-- Runtime stats table
CREATE TABLE rhs.runtime_stats (
    scenario                        TEXT,
    params                          TEXT,
    num_samples                     INTEGER,
    min                             REAL,
    mean                            REAL,
    max                             REAL,
    PRIMARY KEY (scenario, params)
);

-- Insertions

INSERT INTO rhs.runtimes
SELECT runtimes.scenario,runtimes.params,runtimes.runtime
FROM runtimes
LEFT JOIN scenarios
ON runtimes.scenario=scenarios.id
WHERE scenarios.device="2xTahiti";

INSERT INTO rhs.runtime_stats
SELECT
    runtime_stats.scenario,
    runtime_stats.params,
    runtime_stats.num_samples,
    runtime_stats.min,
    runtime_stats.mean,
    runtime_stats.max
FROM runtime_stats
LEFT JOIN scenarios
    ON runtime_stats.scenario=scenarios.id
WHERE scenarios.device="2xTahiti";

INSERT INTO rhs.devices
SELECT *
FROM devices
WHERE id="2xTahiti";

INSERT INTO rhs.scenarios
SELECT * FROM scenarios
WHERE device="2xTahiti";

-- Deletions

DELETE FROM runtimes
WHERE runtimes.scenario IN (
    SELECT id
    FROM scenarios
    WHERE scenarios.device="2xTahiti"
);

DELETE FROM runtime_stats
WHERE runtime_stats.scenario IN (
    SELECT id
    FROM scenarios
    WHERE scenarios.device="2xTahiti"
);

DELETE FROM scenarios
WHERE device="2xTahiti";

DELETE FROM devices WHERE id="2xTahiti";
