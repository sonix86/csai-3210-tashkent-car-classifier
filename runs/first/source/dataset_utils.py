"""Annotation schema and leakage-safe grouping, independent of ML libraries."""
import hashlib
import json
import random
from collections import defaultdict

FIELDS = ["crop", "source", "brand", "colour", "vehicle_id", "photo", "bbox", "evidence"]


def assign_split_groups(rows):
    """Connect all views of a vehicle AND annotations from a single photo."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in rows:
        vehicle, photo = "v:" + row["vehicle_id"], "p:" + row["photo"]
        parent[find(vehicle)] = find(photo)
    members = defaultdict(list)
    for key in parent:
        members[find(key)].append(key)
    names = {root: "group_" + hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()[:16]
             for root, keys in members.items()}
    return [dict(row, source=names[find("v:" + row["vehicle_id"])]) for row in rows]


def grouped_split(rows, seed=42):
    """Keep singleton brands in train; cover validation brands when feasible."""
    effective = assign_split_groups(rows) if all(r.get("vehicle_id") and r.get("photo") for r in rows) else rows
    by_group = defaultdict(list)
    for i, row in enumerate(effective):
        by_group[row["source"]].append(i)
    if len(by_group) < 5:
        raise ValueError("Need at least five independent split groups.")
    brand_groups = defaultdict(set)
    for group, ids in by_group.items():
        for i in ids:
            brand_groups[rows[i]["brand"]].add(group)
    candidates = sorted(by_group)
    random.Random(seed).shuffle(candidates)
    chosen, covered, covered_colours = set(), set(), set()
    target = max(1, round(.2 * len(rows)))
    while candidates and sum(len(by_group[g]) for g in chosen) < target:
        eligible = [g for g in candidates if all(brand_groups[rows[i]["brand"]] - chosen - {g}
                                                for i in by_group[g])]
        if not eligible:
            break
        def priority(g):
            new_brands = len({rows[i]["brand"] for i in by_group[g]} - covered)
            new_colours = len({rows[i].get("colour") for i in by_group[g]} - covered_colours)
            remaining = target - sum(len(by_group[k]) for k in chosen)
            return (new_brands > 0, new_brands / len(by_group[g]), new_colours / len(by_group[g]), new_brands,
                    -abs(remaining - len(by_group[g])))
        group = max(eligible, key=priority)
        chosen.add(group)
        covered.update(rows[i]["brand"] for i in by_group[group])
        covered_colours.update(rows[i].get("colour") for i in by_group[group])
        candidates.remove(group)
    if not chosen:
        raise ValueError("No independent validation group can retain every brand in train. Collect more cars.")
    val = sorted(i for g in chosen for i in by_group[g])
    val_set = set(val)
    return [i for i in range(len(rows)) if i not in val_set], val
