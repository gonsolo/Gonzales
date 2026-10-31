#!/usr/bin/env python3
"""A module may import only from its own layer or the ones below (docs/A5_library_plan.md). Exit status 1 on a violation."""
import glob, os, re, sys

# Lowest first; layers in one tuple are peers and may not import each other.
LAYERS = ["core", "scene/shapes", "scene/appearance", "scene/accel", "scene/description", "transport", "estimators", "paths",
          "integrators", ("cpu", "gpu"), "app"]
IMPORT = re.compile(r"^\s*from\s+gonzales\.([A-Za-z_0-9.]+)\s+import", re.M)


def main():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "gonzales")
    rank = {l: i for i, g in enumerate(LAYERS) for l in ((g,) if isinstance(g, str) else g)}

    def layer_of(path):
        return next((l for l in rank if path == l or path.startswith(l + "/")), None)

    bad, n = [], 0
    for f in glob.glob(os.path.join(src, "**", "*.mojo"), recursive=True):
        rel = os.path.relpath(f, src)[:-5]
        if os.path.basename(rel) == "__init__":
            continue
        n += 1
        layer = layer_of(rel)
        if layer is None:
            bad.append(f"module outside a layer: {rel}")
            continue
        for mod in sorted(set(IMPORT.findall(open(f).read()))):
            other = layer_of(mod.replace(".", "/"))
            if other is None:
                bad.append(f"{rel} imports {mod}, which is in no layer")
            elif rank[other] > rank[layer] or (other != layer and rank[other] == rank[layer]):
                bad.append(f"upward import: {rel} -> {mod}")
    print(f"{n} modules in {len(rank)} layers")
    for b in sorted(bad):
        print(b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
