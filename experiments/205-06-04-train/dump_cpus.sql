--
-- Create a database 2xTahiti.db and move any 2xTahiti results into
-- it.
--

DELETE FROM devices WHERE type='2';

-- Purge from kernels table forward.
DELETE FROM scenarios      WHERE device NOT IN (SELECT id FROM devices);
DELETE FROM kernels        WHERE id NOT IN (SELECT DISTINCT kernel FROM scenarios);
DELETE FROM kernel_names   WHERE id NOT IN (SELECT DISTINCT kernel FROM scenarios);
DELETE FROM datasets       WHERE id NOT IN (SELECT DISTINCT dataset FROM scenarios);
DELETE FROM runtimes       WHERE scenario NOT IN (SELECT id FROM scenarios);
DELETE FROM runtime_stats  WHERE scenario NOT IN (SELECT id FROM scenarios);
DELETE FROM oracle_params  WHERE scenario NOT IN (SELECT id FROM scenarios);
