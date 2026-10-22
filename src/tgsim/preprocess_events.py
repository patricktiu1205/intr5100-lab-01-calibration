"""TGSIM extension: prepare HV-HV / HV-AV events.

Splits each continuous pair into events at stop-and-go waves: a stop run is a
maximal interval of follower speed_kf < STOP_V lasting at least STOP_MIN s;
event boundaries are stop-run end times. Emits per-event tables time,fv,lv,dhw
(seconds, m, m/s) under data/tgsim/<pair>/ and a JSON manifest.

IDM spacing for these data is s = projected gap - leader length (xloc is the
front-bumper position); dhw is written as s + leader length so the table keeps
the raw distance-headway meaning.
"""
import csv, json, math, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TG   = os.path.join(ROOT, "data", "raw", "tgsim")
OUT  = os.path.join(ROOT, "data", "tgsim")
STOP_V, STOP_MIN, MIN_EVENT_PTS = 0.3, 1.0, 50

PAIRS = {"HV-HV": "HV-HV.csv", "HV-AV": "HV-AV.csv"}


def load_pair(path):
    rows = list(csv.DictReader(open(path, newline="")))
    fol, lead = {}, {}
    meta = {}
    for r in rows:
        t = round(float(r["time"]), 2)
        rec = dict(x=float(r["xloc_kf"]), y=float(r["yloc_kf"]),
                   v=float(r["speed_kf"]), a=float(r["acceleration_kf"]),
                   length=float(r["length_smoothed"]))
        if r["role"] == "follower":
            fol[t] = rec
        else:
            lead[t] = rec
    return fol, lead


def stop_runs(fol):
    ts = sorted(fol)
    runs, cur = [], None
    for t in ts:
        if fol[t]["v"] < STOP_V:
            cur = [t, t] if cur is None else [cur[0], t]
        else:
            if cur and cur[1] - cur[0] >= STOP_MIN:
                runs.append(tuple(cur))
            cur = None
    if cur and cur[1] - cur[0] >= STOP_MIN:
        runs.append(tuple(cur))
    return runs


def main():
    manifest = {}
    for pair, fname in PAIRS.items():
        fol, lead = load_pair(os.path.join(TG, fname))
        ts = sorted(fol)
        runs = stop_runs(fol)
        # boundaries: end of each stop run
        bounds = [e for _, e in runs]
        outdir = os.path.join(OUT, pair)
        os.makedirs(outdir, exist_ok=True)
        events, start = [], ts[0]
        # if data begin inside a stop, start at first moving sample
        if fol[start]["v"] < STOP_V and runs and runs[0][0] == start:
            start = next(t for t in ts if t > runs[0][1] - 0.01)
        for b in bounds + [ts[-1] + 1]:
            seg = [t for t in ts if start - 1e-9 <= t < b - 1e-9]
            if len(seg) >= MIN_EVENT_PTS:
                events.append(seg)
            start = b
        kept = []
        for i, seg in enumerate(events, 1):
            l0 = lead[seg[0]]["length"]
            f0 = fol[seg[0]]["length"]
            rows_out = []
            for t in seg:
                f, l = fol[t], lead[t]
                gap = math.hypot(l["x"] - f["x"], l["y"] - f["y"])
                rows_out.append((t - seg[0], f["v"], l["v"], gap))
            name = f"event_{i}.csv"
            tmp = os.path.join(outdir, name + ".tmp")
            with open(tmp, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time", "fv", "lv", "dhw"])
                w.writerows(rows_out)
            os.replace(tmp, os.path.join(outdir, name))
            kept.append(dict(event=name, n=len(seg),
                             t0=seg[0], t1=seg[-1],
                             s_min=round(min(r[3] for r in rows_out) - l0, 2),
                             leader_length=l0, follower_length=f0))
        manifest[pair] = dict(source=fname, stop_runs=len(runs),
                              events=kept,
                              total_samples=sum(e["n"] for e in kept),
                              spacing_definition=(
                                  "s = hypot(leader xloc/yloc - follower xloc/yloc) - leader_length "
                                  "(metric m; dhw column = raw centre distance)"))
        print(pair, "stop runs", len(runs), "-> events", len(kept),
              "samples", manifest[pair]["total_samples"])
        for e in kept:
            print("  ", e["event"], e["n"], f"t {e['t0']}-{e['t1']}",
                  "s_min", e["s_min"])
    tmp = os.path.join(OUT, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, os.path.join(OUT, "manifest.json"))


if __name__ == "__main__":
    main()
