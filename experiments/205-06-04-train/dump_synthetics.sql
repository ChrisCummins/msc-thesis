--
-- Dump all results from synthetic benchmarks.
--

DELETE FROM kernel_names WHERE synthetic=1;

-- Purge from kernels table forward.
DELETE FROM kernels       WHERE id NOT IN (SELECT DISTINCT id FROM kernel_names);
DELETE FROM scenarios     WHERE kernel NOT IN (SELECT id FROM kernels);
DELETE FROM runtimes      WHERE scenario NOT IN (SELECT id FROM scenarios);
DELETE FROM runtime_stats WHERE scenario NOT IN (SELECT id FROM scenarios);
DELETE FROM oracle_params  WHERE scenario NOT IN (SELECT id FROM scenarios);
