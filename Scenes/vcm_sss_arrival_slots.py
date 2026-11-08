#!/usr/bin/env python3
"""Do VCM's strategies of a path through skin exist exactly when its weights count them, near the light cache's limit?

Models the slot bookkeeping of integrators/vcm/light.mojo and camera.mojo: a light path stores at most MAX_VERTS slots
(an area light's origin, non-delta vertices, a skin arrival before each walk, rough skin's lobe vertex, the walk's
exit); VcmStrategies.past_cache calls a split long when its light non-delta count (the slot index) plus the camera's
non-delta count exceeds MAX_STORED, and then only the camera strategies (hit, NEE) exist. Every split of a path must
classify it alike, and when it is not long every structurally possible strategy, the exit merge included, must run.
Enumerates every path of diffuse (D), glass (G), smooth (S) and rough (R) skin walks, smooth (F) and rough (Q) skin\nreflections, both light kinds.
Rules: "current" (dffec8af), "fixed": the camera counts one cache slot per skin hit from outside, the arrival a light
path stores there; the light stores an arrival whenever a slot is left and ends its path when that was the last one.
"deep": "fixed", and the camera goes on past the cache with its own strategies only, so every path is reached.
Usage: vcm_sss_arrival_slots.py    (exit status 1 on failure)
"""
import itertools, sys

MAX_VERTS = 10
MAX_STORED = MAX_VERTS - 1


def expand(tokens):
    """Path vertices light -> camera: (kind, walk id, role). A walk is two vertices, a (light entry, camera exit) and b."""
    v = [("light", None, None)]
    for w, t in enumerate(tokens):
        if t in "DGFQ":
            v.append((t, None, None))
        else:
            v += [(t, w, "a"), (t, w, "b")]
    return v + [("camera", None, None)]


def light_slots(path, area, rule):
    """{path index: [(slot kind, local)]} of what the light path stores, as light.mojo does."""
    slots, n = {}, 0
    def store(i, kind):
        nonlocal n
        slots.setdefault(i, []).append((kind, n)); n += 1
    if area:
        store(0, "origin")
    i = 1
    while i < len(path) - 1:
        kind, _, role = path[i]
        if n >= MAX_VERTS or (n if area else n + 1) > MAX_STORED:
            break                                          # top of _bdpt_light_path_bounce
        if kind == "D":
            store(i, "vertex")
        elif kind in "FQ":                                 # skin that reflects: the arrival is stored before the choice
            need = (2 if kind == "Q" else 1) if rule == "current" else 1
            if n + need <= MAX_VERTS:
                store(i, "arrival")
            if rule != "current" and n >= MAX_VERTS:
                break
            if kind == "Q":
                if n >= MAX_VERTS:
                    break
                store(i, "vertex")
        elif kind in "SR":                                 # role a: the light enters here
            need = (3 if kind == "R" else 2) if rule == "current" else 1
            if n + need <= MAX_VERTS:
                store(i, "arrival")
            if rule != "current" and n >= MAX_VERTS:
                break
            if kind == "R":
                if n >= MAX_VERTS:
                    break
                store(i, "vertex")                         # rough skin's lobe vertex at the entry
            if n >= MAX_VERTS:
                break                                      # the walk exit's slot guard
            store(i + 1, "exit")
            i += 1
        i += 1
    return slots


def camera_vertices(path, rule):
    """({path index: cam_nd} for connections, the same for merges, lowest index reached, final cam_nd), as camera.mojo
    counts. Under
    "fixed" every skin hit from outside counts the arrival a light path stores there: for a walk at its exit (where the
    light arrives), before the exit merges; for a reflection after the vertex merges (the light's own slot there already
    sits after that arrival) and before it connects."""
    out, outm, n_verts, arr = {}, {}, 0, 0
    fixed = rule != "current"
    cap = MAX_STORED if rule != "deep" else 10 ** 9      # the camera's own cut-off; deep has only the depth limit
    i = len(path) - 2
    while i >= 1:
        kind, _, role = path[i]
        if n_verts >= cap + 1:
            return out, outm, i + 1, n_verts + arr
        if kind in "DQ":
            if n_verts >= cap:
                return out, outm, i + 1, n_verts + arr                    # ends_at_vertex
            n_verts += 1; outm[i] = n_verts + arr
            if kind == "Q" and fixed:
                arr += 1
            out[i] = n_verts + arr
        elif kind == "F":
            arr += fixed
        elif kind in "SR":                                 # role b: the camera enters here
            if kind == "R":
                if n_verts >= cap:
                    return out, outm, i + 1, n_verts + arr
                n_verts += 1; out[i] = outm[i] = n_verts + arr
            if n_verts >= cap:
                return out, outm, i + 1, n_verts + arr
            n_verts += 1; arr += fixed
            out[i - 1] = outm[i - 1] = n_verts + arr       # the walk's exit, at role a
            i -= 1
        i -= 1
    return out, outm, 0, n_verts + arr


def nd_of(local, area): return local if area else local + 1


def check(tokens, area, rule):
    path = expand(tokens)
    n = len(path) - 1
    ls = light_slots(path, area, rule)
    cv, cvm, reached, cam_final = camera_vertices(path, rule)
    views, ran, possible = [], set(), set()
    def slot(i, kinds):
        k = [x for x in ls.get(i, []) if x[0] in kinds]
        return k[0] if k else None
    for i in range(n):                                     # connections: light vertex i, camera vertex i + 1
        j = i + 1
        if path[i][0] in "GF" or path[j][0] in "GF" or path[i][2] == "a" or path[j][2] == "b":
            continue                                       # delta, a skin arrival, or across a walk
        possible.add(("c", i))
        if j == n:                                         # light trace to the lens
            sl = slot(i, ("origin", "vertex", "exit"))
            if sl is not None and nd_of(sl[1], area) <= MAX_STORED:
                ran.add(("c", i)); views.append(False)
            continue
        if j not in cv:
            continue
        if i == 0:                                         # NEE: always runs; long by the camera's count alone
            ran.add(("c", i)); views.append(cv[j] > MAX_STORED); continue
        sl = slot(i, ("vertex", "exit"))
        if sl is not None and nd_of(sl[1], area) + cv[j] <= MAX_STORED:
            ran.add(("c", i)); views.append(False)
    for k in range(1, n):                                  # merges: an ordinary vertex, or a walk exit with an arrival
        kind, _, role = path[k]
        if not (kind in "DQ" or (kind in "SR" and role == "a")):
            continue
        possible.add(("m", k))
        sl = slot(k, ("vertex",) if kind in "DQ" else ("arrival",))
        if sl is not None and k in cvm and nd_of(sl[1], area) + cvm[k] - 1 <= MAX_STORED:
            ran.add(("m", k)); views.append(False)
    possible.add(("hit",))                                 # the camera hits the light: long by its whole count
    if reached <= 1:
        ran.add(("hit",)); views.append(cam_final > MAX_STORED)
    if not views:                                          # nothing reaches this path: energy lost
        return "unreached" if rule == "deep" else None
    if len(set(views)) > 1:
        return "splits disagree"
    if not views[0]:
        missing = sorted(possible - ran)
        if missing:
            return "missing " + ",".join(str(m) for m in missing)
    return None


def main():
    worst = {}
    for rule in ("current", "fixed", "deep"):
        bad, total, example = 0, 0, None
        for L in range(1, 7):
            for tokens in itertools.product("DSRGFQ", repeat=L):
                if tokens.count("G") + tokens.count("F") > 1:
                    continue
                for area in (True, False):
                    total += 1
                    r = check(tokens, area, rule)
                    if r:
                        bad += 1
                        example = example or ("".join(tokens), "area" if area else "point", r)
        worst[rule] = bad
        print(f"  {rule:8s} {bad:6d} of {total} paths with a strategy counted but not run, or splits that disagree"
              + (f"; e.g. {example}" if example else ""))
    sys.exit(0 if worst["fixed"] == 0 and worst["deep"] == 0 and worst["current"] > 0 else 1)


if __name__ == "__main__":
    main()
