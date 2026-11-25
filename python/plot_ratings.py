"""Plot the ratings saved by `play evaluate_multi` against training progress.

usage: python plot_ratings.py ../ratings/rating_M.txt [more rating files ...] [-o out.png] [--show] [--ci]

A rating file has one "model rating games" line per model. The x axis is the number of self-play games in the model
name (M_12b_60480B.pt -> 60480). Models are grouped into one line per run (the name without its game count), so a
file holding several runs, or several files, give several lines.
With --ci and an existing <rating file>.pairs, the 95% confidence interval of each rating is drawn too, from a
Bradley-Terry fit of those pair results. It is relative to the run's first model (fixed at its saved rating), so it
mostly shows how uncertain the total progress is, not each step between neighbouring models.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from select_models import fit_bradley_terry

# same reference palette as plot_training.py : categorical order blue, orange, aqua; muted ink for annotations
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

NAME_RE = re.compile(r"^(.*?)_?(\d+)B?\.pt$")


def run_and_games(name):
    m = NAME_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def read_ratings(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], float(parts[1]), int(parts[2]) if len(parts) > 2 else 0))
    return rows


def read_ci(path):
    """name -> 95% half-width in Elo, from <path>.pairs. Empty if there is no pairs file."""
    pairs = Path(str(path) + ".pairs")
    if not pairs.exists():
        return {}
    matches = []
    for line in pairs.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4:
            matches.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    if not matches:
        return {}
    names, _, ses, _ = fit_bradley_terry(matches, [], 1.0, None)
    return {n: 1.96 * se for n, se in zip(names, ses)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ratings", nargs="+", type=Path, help="rating files written by evaluate_multi")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output image (default: <first rating file>.png)")
    ap.add_argument("--show", action="store_true", help="also open a window")
    ap.add_argument("--ci", action="store_true", help="draw 95%% confidence bands (needs <rating file>.pairs)")
    args = ap.parse_args()

    runs = {}  # run name -> list of (games, rating, ci, model)
    skipped = []
    for path in args.ratings:
        ci = read_ci(path) if args.ci else {}
        for name, rating, _ in read_ratings(path):
            run, games = run_and_games(name)
            if run is None:
                skipped.append(name)
                continue
            runs.setdefault(run, []).append((games, rating, ci.get(name), name))
    if skipped:
        print("no game count in the name, skipped : " + " ".join(skipped))
    if not runs:
        raise SystemExit("no ratings found")
    if len(runs) > len(SERIES):
        raise SystemExit(f"{len(runs)} runs in one chart is too many to tell apart; plot at most {len(SERIES)}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for color, (run, pts) in zip(SERIES, sorted(runs.items())):
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        if any(p[2] is not None for p in pts):
            lo = [p[1] - (p[2] or 0.0) for p in pts]
            hi = [p[1] + (p[2] or 0.0) for p in pts]
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, label=run, zorder=3)

    has_ci = any(p[2] is not None for pts in runs.values() for p in pts)
    # title = "Elo rating by training progress"
    # if len(runs) == 1:  # a single line has no legend : the title names it
    #     title += f" : {next(iter(runs))}"
    # if has_ci:
    #     title += " (band : 95% CI)"
    # ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_title("")
    ax.set_xlabel("self-play games", fontsize=15, color=MUTED, fontweight='bold')
    ax.set_ylabel("Elo rating", fontsize=15, color=MUTED, fontweight='bold')
    ax.tick_params(labelsize=15, colors=MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if len(runs) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.margins(x=0.08)

    fig.tight_layout()
    out = args.out or args.ratings[0].with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
