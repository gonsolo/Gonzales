# One progress line for every renderer.
#
# The path tracer's tile loop and its two GPU sample loops each timed
# themselves, printed `progress_str` with a carriage return and wrote their
# own "Done" line; VCM and SPPM printed nothing at all, so a ten-minute
# render sat silent. `Progress` is that pattern once:
#
#     var prog = Progress(n_spp, "spp", quiet=verbose)
#     for si in range(n_spp):
#         ...                     # enqueue the pass
#         ctx.synchronize()       # GPU drivers: report finished work, not queued
#         prog.update(si + 1)
#     var seconds = prog.finish()
#
# The synchronize stays with the caller: only a driver knows its device
# context, and a CPU loop has nothing to wait for.

from std.time import perf_counter_ns


def _fmt_f1(v: Float64) -> String:
    var i = Int(v)
    var frac = Int((v - Float64(i)) * 10.0 + 0.5)
    if frac >= 10:
        i += 1; frac = 0
    return String(i) + "." + String(frac)


def fmt_time(s: Float64) -> String:
    var sec = Int(s)
    var min = sec // 60
    var rem = sec % 60
    if min > 0:
        var rs = String(rem)
        if rem < 10: rs = "0" + rs
        return String(min) + "m " + rs + "s"
    return _fmt_f1(s) + "s"


def progress_str(done: Int, total: Int, elapsed: Float64, unit: String) -> String:
    var pct = _fmt_f1(Float64(done) * 100.0 / Float64(total))
    var est = Float64(0.0)
    if done > 0:
        est = elapsed * Float64(total) / Float64(done)
    return ("Rendering: " + String(done) + " / " + String(total)
        + " " + unit + " (" + pct + "%) | Elapsed: " + fmt_time(elapsed)
        + " | Total Est.: " + fmt_time(est) + "                ")


struct Progress(Movable):
    """A render's progress line: `update` rewrites it in place, `finish`
    prints the final "Done" line and returns the elapsed seconds. `quiet`
    silences both (the timing still works), for callers that print their own
    per-pass lines."""
    var total: Int
    var unit: String
    var t0: Int
    var quiet: Bool

    def __init__(out self, total: Int, unit: String, quiet: Bool = False):
        self.total = max(total, 1)
        self.unit = unit
        self.t0 = perf_counter_ns()
        self.quiet = quiet

    def elapsed(self) -> Float64:
        return Float64(perf_counter_ns() - self.t0) / 1.0e9

    def update(self, done: Int):
        if not self.quiet:
            print(progress_str(done, self.total, self.elapsed(), self.unit), end="\r")

    def finish(self) -> Float64:
        var s = self.elapsed()
        if not self.quiet:
            print("Rendering: " + String(self.total) + " / " + String(self.total)
                + " " + self.unit + " (100.0%) | Done: " + fmt_time(s) + "                ")
        return s
