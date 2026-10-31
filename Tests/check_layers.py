#!/usr/bin/env python3
"""A module may import only from its own layer or the ones below (docs/A5_library_plan.md). Exit status 1 on a new violation."""
import glob, os, re, sys

# Lowest first; layers in one tuple are peers and may not import each other.
LAYERS = ["core", "scene", "transport", "estimators", "paths", "integrators", ("cpu", "gpu"), "app"]
# Today's debts. The list only shrinks: docs/A5_library_plan.md, Order.
ALLOWED = {"shade_context>guide", "shade_context>restir_gi"}
IMPORT = re.compile(r"^\s*from\s+gonzales\.([a-z]+)\.([A-Za-z_0-9]+)\s+import", re.M)


def main():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "gonzales")
    rank = {l: i for i, g in enumerate(LAYERS) for l in ((g,) if isinstance(g, str) else g)}
    bad, debts, n = [], set(), 0
    for layer in rank:
        for f in glob.glob(os.path.join(src, layer, "*.mojo")):
            m = os.path.basename(f)[:-5]
            if m == "__init__":
                continue
            n += 1
            for l, x in sorted(set(IMPORT.findall(open(f).read()))):
                if rank[l] > rank[layer] or (l != layer and rank[l] == rank[layer]):
                    edge = f"{m}>{x}"
                    debts.add(edge)
                    if edge not in ALLOWED:
                        bad.append(f"upward import: {layer}/{m} -> {l}/{x}")
    stray = [os.path.basename(f) for f in glob.glob(os.path.join(src, "*.mojo")) if not f.endswith("__init__.mojo")]
    bad += [f"module outside a layer: {f}" for f in stray]
    bad += [f"paid debt still listed: {e}" for e in sorted(ALLOWED - debts)]
    print(f"{n} modules in {len(rank)} layers, {len(debts)} upward imports allowed for now")
    for b in bad:
        print(b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
