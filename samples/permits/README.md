# Permit applications — a real municipal event log (300 cases)

The first 300 cases of **"Receipt phase of an environmental permit application
process (WABO), CoSeLoG project"** (J.C.A.M. Buijs, Eindhoven University of
Technology), published on 4TU.ResearchData. Taken from the copy distributed with
pm4py's test data and converted with `tools/xes_to_csv.py --max-cases 300`
(completed events only).

A real Dutch municipality's intake process: 300 applications, 1,725 events,
24 activities, 40+ staff. No configuration is needed — the shape is detected:

    python3 run_tabular.py --file samples/permits/permits.csv --entity application

It is the engine's held-out check on a log it was never tuned on. It found the
intake route (T02 → T04 → T05 → T06 → T10) and the rework loop, and it exposed a
ranking flaw: a slow detour taken by 13 of 300 cases had been headlined as
"83% of the time". Waits are now ranked by the total time they cost across
every case (`induction/findings.py`), with a test pinning it.
