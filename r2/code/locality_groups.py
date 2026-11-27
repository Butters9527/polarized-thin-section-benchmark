# -*- coding: utf-8 -*-
"""Rebuild the rock-name + locality groups used by the locality-grouped protocol and check them.

Sections are linked when they share a rock name (type_key) or a locality; the groups are the
connected components of this graph (77 groups for the 321 usable NJU sections). The localities
were taken from the information tables distributed with the NJU dataset.

usage: python locality_groups.py --index ../data
Checks that the groups recomputed from section_locality.csv equal the stored loc_group column and
that, in folds.csv (protocol = locality), no group is split between the test fold and the rest.
"""
import argparse
import csv
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="../data")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.index, "section_locality.csv"), encoding="utf-8-sig")))
    parent = {r["section_key"]: r["section_key"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for field in ("type_key", "locality"):
        first = {}
        for r in rows:
            k = r[field]
            if k in first:
                parent[find(r["section_key"])] = find(first[k])
            else:
                first[k] = r["section_key"]
    comp = defaultdict(set)
    for r in rows:
        comp[find(r["section_key"])].add(r["section_key"])
    stored = defaultdict(set)
    for r in rows:
        stored[r["loc_group"]].add(r["section_key"])
    same = sorted(map(sorted, comp.values())) == sorted(map(sorted, stored.values()))
    print(f"sections: {len(rows)}; recomputed groups: {len(comp)}; stored groups: {len(stored)}; identical: {same}")

    grp = {r["section_key"]: r["loc_group"] for r in rows}
    role = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(a.index, "folds.csv"), encoding="utf-8-sig")):
        if r["protocol"] == "locality":
            role[(int(r["rep"]), int(r["fold"]))][r["section_key"]] = r["role"]
    leaks = 0
    for key, m in role.items():
        test_g = {grp[s] for s, ro in m.items() if ro == "test"}
        rest_g = {grp[s] for s, ro in m.items() if ro != "test"}
        leaks += len(test_g & rest_g)
    print(f"locality folds: {len(role)}; groups shared between test and training/early-stopping: {leaks}")


if __name__ == "__main__":
    main()
