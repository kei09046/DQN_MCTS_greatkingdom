"""Plot the Elo curves of several training runs (e.g. the full method and its ablations) on one chart.

usage: python plot_ablation.py ../ratings/rating_M.txt ../ratings/rating_MnS.txt[:label] [...]
                               [-o out.png] [--show] [--logx] [--diff] [--from GAMES] [--xmax GAMES | common]

Each argument is a rating file written by `play evaluate_multi` ("model rating games" per line), optionally followed by
":label" for the legend (default : the run name in the model file names, e.g. MnS). One line per run; a file holding
several runs gives several lines. The x axis is the number of self-play games in the model name.

--logx        log scale x axis (as in KataGo's ablation plots).
--diff        add a lower panel : each run's rating minus the first run's, the first run's rating being linearly
              interpolated at the other run's checkpoints (only inside the first run's range).
--from GAMES  print the mean difference to the first run over checkpoints with at least GAMES self-play games.
--xmax        drop checkpoints past GAMES self-play games; "common" : past the end of the shortest run, so that
              only the range every run covers is shown.

Note : ratings of different files are only comparable if they share one rating scale. A file fitted on its own is
anchored at its own first model (usually rating 0) and inflated by games within its own lineage, so two separate files
are aligned only at that anchor. For a fair comparison, rate all runs into one file (evaluate_multi with every run's
checkpoints and the same rating path) so that cross-run games tie the scales together. The script warns when the
runs come from separate files.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# reference palette of plot_ratings.py, extended for more runs
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c9a227"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

NAME_RE = re.compile(r"^(.*?)_(\d+)b_(\d+)B?\.pt$")


def run_and_games(name):
    m = NAME_RE.match(Path(name).name)
    return (m.group(1), int(m.group(3))) if m else (None, None)


def read_ratings(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], float(parts[1])))
    return rows


def parse_arg(arg):
    """"file[:label]" -> (Path, label or None). A ':' inside a Windows drive letter is not a label separator."""
    head, sep, tail = arg.rpartition(":")
    if sep and head and not Path(arg).exists() and "/" not in tail and "\\" not in tail:
        return Path(head), tail
    return Path(arg), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ratings", nargs="+", help="rating files, each optionally followed by :label")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output image (default: ablation.png next to the first file)")
    ap.add_argument("--show", action="store_true", help="also open a window")
    ap.add_argument("--logx", action="store_true", help="log scale x axis")
    ap.add_argument("--diff", action="store_true", help="add a panel with each run's rating minus the first run's")
    ap.add_argument("--from", dest="start", type=int, default=None, metavar="GAMES",
                    help="print the mean difference to the first run from this many self-play games on")
    ap.add_argument("--xmax", default=None, metavar="GAMES",
                    help='last self-play game count shown, or "common" for the end of the shortest run')
    args = ap.parse_args()

    runs = {}  # (file, run name) -> {"label", "pts": [(games, rating)]}, in argument order
    skipped = []
    for arg in args.ratings:
        path, label = parse_arg(arg)
        if not path.exists():
            raise SystemExit(f"no such file : {path}")
        for name, rating in read_ratings(path):
            run, games = run_and_games(name)
            if run is None:
                skipped.append(name)
                continue
            entry = runs.setdefault((path, run), {"label": label or run, "pts": []})
            entry["pts"].append((games, rating))
    if skipped:
        print("no game count in the name, skipped : " + " ".join(skipped))
    if not runs:
        raise SystemExit("no ratings found")
    if len(runs) > len(SERIES):
        raise SystemExit(f"{len(runs)} runs in one chart is too many to tell apart; plot at most {len(SERIES)}")
    files = {f for f, _ in runs}
    if len(files) > 1:
        print("warning : runs come from separate rating files, so their scales are only aligned at each file's anchor "
              "model. Rate all runs into one file for a fair comparison.")

    for entry in runs.values():
        entry["pts"].sort()
    if args.xmax is not None:
        xmax = min(e["pts"][-1][0] for e in runs.values()) if args.xmax == "common" else int(args.xmax)
        for entry in runs.values():
            entry["pts"] = [p for p in entry["pts"] if p[0] <= xmax]
        runs = {k: e for k, e in runs.items() if e["pts"]}
        print(f"showing checkpoints up to {xmax} games")
    base = next(iter(runs.values()))
    bx = np.array([p[0] for p in base["pts"]], dtype=float)
    by = np.array([p[1] for p in base["pts"]])

    if args.diff:
        fig, (ax, axd) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        axd = None

    for color, entry in zip(SERIES, runs.values()):
        x = np.array([p[0] for p in entry["pts"]], dtype=float)
        y = np.array([p[1] for p in entry["pts"]])
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, label=entry["label"], zorder=3)
        print(f"{entry['label']} : {len(x)} checkpoints, {x[0]:.0f} .. {x[-1]:.0f} games, "
              f"final rating {y[-1]:.0f}")

        if entry is base:
            continue
        inside = (x >= bx[0]) & (x <= bx[-1])
        d = y[inside] - np.interp(x[inside], bx, by)
        if axd is not None and inside.any():
            axd.plot(x[inside], d, color=color, linewidth=2, marker="o", markersize=5,
                     markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        if args.start is not None:
            late = x[inside] >= args.start
            if late.any():
                print(f"  {entry['label']} - {base['label']} from {args.start} games : mean {d[late].mean():+.0f} Elo "
                      f"over {late.sum()} checkpoints")
            else:
                print(f"  {entry['label']} - {base['label']} from {args.start} games : no checkpoints in range")

    axes = [ax] + ([axd] if axd is not None else [])
    for a in axes:
        if args.logx:
            a.set_xscale("log")
        a.tick_params(labelsize=15, colors=MUTED)
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
        a.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        if args.start is not None:
            a.axvline(args.start, color=MUTED, linewidth=1, linestyle=":", zorder=2)
        a.margins(x=0.05)
    ax.set_ylabel("Elo rating", fontsize=15, color=MUTED, fontweight="bold")
    if axd is not None:
        axd.axhline(0.0, color=MUTED, linewidth=1, zorder=2)
        axd.set_ylabel(f"− {base['label']}", fontsize=15, color=MUTED, fontweight="bold")
    axes[-1].set_xlabel("self-play games", fontsize=15, color=MUTED, fontweight="bold")
    ax.legend(frameon=False, fontsize=12, loc="upper left")

    fig.tight_layout()
    first = parse_arg(args.ratings[0])[0]
    out = args.out or first.with_name("ablation.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
