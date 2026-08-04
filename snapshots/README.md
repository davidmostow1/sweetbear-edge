# Bear Edge state snapshots

`bear_edge_state.json` is the most recent automated capture of
`davidmostow1/bear-edge-platform`, produced by
`scripts/bear_edge_snapshot.py`. A dated copy is kept alongside it whenever
something material changes, so the history of the audit's standing findings is
version-controlled rather than living in a chat log or a Drive folder.

State is stored here, in git, rather than in external storage on purpose: the
provenance question this whole audit keeps returning to is "can an outside
reviewer verify this claim." A JSON file committed next to the script that
produced it, at a known commit, is verifiable. A file in someone's Drive is not.

Do not hand-edit these files. If a value looks wrong, fix the script and re-run
it, so the record and the tool that made it stay in agreement.
