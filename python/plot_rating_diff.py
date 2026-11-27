"""Plot the Elo difference between two runs, from `play evaluate_two` logs, against training progress.

usage: python plot_rating_diff.py ../ratings/MvsMnTnP.txt [../ratings/MvsMnT.txt ...] [-o out.png] [--show]
                                  [--smooth K] [--from GAMES] [--target {first,second}]
                                  [--min-games GAMES] [--band]

Each log holds one evaluate_two result per checkpoint pair : two "Model loaded: <path> (...)" lines, then
"win count : <wins>" and the first model's win rate. The difference is the target model's Elo minus the baseline's,
400 * log10(p / (1 - p)) with p the target's win rate, so above 0 means the target is stronger. By default the second
model is the target and the first the baseline (MvsMnT.txt -> MnT − M); --target first swaps that. The x axis is the
number of self-play games in the target model's name (MnT_12b_11520B.pt -> 11520). The printed CI (and the band,
with --band) is the 95% Wilson interval of the win rate, mapped to Elo. One line per file, so several comparisons (e.g. MvsMnR.txt MvsMnT.txt
MvsMnTnP.txt) can share one chart; when two files give the same "target − baseline" label, the file name is added.

--target T   which model of each pair is the target : "second" (default) or "first".
--min-games GAMES  drop checkpoints with fewer self-play games, before smoothing and pooling (early ones are noisy).
--band       draw the 95% interval as a shaded band (hidden by default).

--smooth K   pool the games of K neighbouring checkpoints (centred window) for each point : K = 3 with 96 games per
             checkpoint narrows the band from about +-70 to +-40 Elo. Points are drawn at the centre checkpoint.
--from GAMES also print the pooled difference over checkpoints with at least GAMES self-play games (e.g. the late
             stage only), and mark that point with a dotted line.
"""
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

# plot_ratings.py's three hues, extended in fixed order (checked with the dataviz palette validator)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#f2a600", "#ee7fac", "#008a17"]
MARKERS = ["o", "s", "^", "D", "v", "P"]  # second cue besides colour
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LOADED_RE = re.compile(r"Model loaded:\s*(\S+)")
WINS_RE = re.compile(r"win count\s*:\s*(\d+)")
RATE_RE = re.compile(r"^\s*([01](?:\.\d+)?|\.\d+)\s*$")
NAME_RE = re.compile(r"^(.*?)_(\d+)b_(\d+)B?\.pt$")
Z = 1.96


def model_info(path):
    """(run name, self-play games) from a model path, e.g. ../testmodels/MnT_12b_10560B.pt -> ("MnT", 10560)."""
    m = NAME_RE.match(Path(path).name)
    return (m.group(1), int(m.group(3))) if m else (None, None)


def read_log(path):
    """list of (games, first model, second model, wins, total games) in file order."""
    results = []
    models, wins = [], None
    for line in Path(path).read_text().splitlines():
        if (m := LOADED_RE.search(line)):
            if len(models) == 2:  # a new comparison started before the previous one finished
                models, wins = [], None
            models.append(m.group(1))
        elif (m := WINS_RE.search(line)):
            wins = int(m.group(1))
        elif wins is not None and (m := RATE_RE.match(line)):
            rate = float(m.group(1))
            if len(models) == 2 and rate > 0.0:
                total = round(wins / rate)
                games = model_info(models[0])[1]
                if games is not None:
                    results.append((games, models[0], models[1], wins, total))
            models, wins = [], None
    return results


def elo(p):
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return 400.0 * math.log10(p / (1.0 - p))


def wilson(wins, n):
    p = wins / n
    denom = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre - half, centre + half


def smooth(rows, k):
    """pool wins / totals of k neighbouring checkpoints (centred, shrunk at the ends)."""
    if k <= 1:
        return rows
    half = k // 2
    out = []
    for i, (games, m1, m2, _, _) in enumerate(rows):
        window = rows[max(0, i - half): i + half + 1]
        out.append((games, m1, m2, sum(r[3] for r in window), sum(r[4] for r in window)))
    return out


def pooled(rows):
    wins, total = sum(r[3] for r in rows), sum(r[4] for r in rows)
    lo, hi = wilson(wins, total)
    return wins, total, elo(wins / total), elo(lo), elo(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path, help="evaluate_two logs, e.g. ../ratings/MvsMnTnP.txt")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output image (default: <first log>.png)")
    ap.add_argument("--show", action="store_true", help="also open a window")
    ap.add_argument("--smooth", type=int, default=1, metavar="K", help="pool K neighbouring checkpoints per point")
    ap.add_argument("--from", dest="start", type=int, default=None, metavar="GAMES",
                    help="also report the pooled difference from this many self-play games on")
    ap.add_argument("--min-games", type=int, default=0, metavar="GAMES",
                    help="ignore checkpoints with fewer self-play games than this")
    ap.add_argument("--band", action="store_true", help="draw the 95%% interval band")
    ap.add_argument("--target", choices=("first", "second"), default="second",
                    help="which model of each pair the difference is measured for (default: second)")
    args = ap.parse_args()
    if args.smooth < 1:
        raise SystemExit("--smooth must be at least 1")
    if len(args.logs) > len(SERIES):
        raise SystemExit(f"{len(args.logs)} logs in one chart is too many to tell apart; plot at most {len(SERIES)}")

    series = []
    for path in args.logs:
        rows = read_log(path)
        if args.target == "second":  # second model's view : its wins are the first model's losses
            rows = [(model_info(m2)[1] or games, m2, m1, n - w, n) for games, m1, m2, w, n in rows]
        rows = sorted(r for r in rows if r[0] >= args.min_games)
        if not rows:
            print(f"{path} : no results found")
            continue
        target, baseline = model_info(rows[0][1])[0], model_info(rows[0][2])[0]
        series.append((path, rows, f"{target} − {baseline}"))
    labels = [label for _, _, label in series]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for color, marker, (path, rows, label) in zip(SERIES, MARKERS, series):
        if labels.count(label) > 1:
            label = f"{label} ({path.stem})"
        shown = smooth(rows, args.smooth)
        x = [r[0] for r in shown]
        y = [elo(r[3] / r[4]) for r in shown]
        if args.band:
            lo, hi = zip(*(wilson(r[3], r[4]) for r in shown))
            ax.fill_between(x, [elo(p) for p in lo], [elo(p) for p in hi], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)

        wins, total, d, dlo, dhi = pooled(rows)
        print(f"{path.name} : {label}, {len(rows)} checkpoints, pooled {wins}/{total} = {wins / total:.3f} "
              f"-> {d:+.0f} Elo (95% CI {dlo:+.0f} .. {dhi:+.0f})")
        if args.start is not None:
            late = [r for r in rows if r[0] >= args.start]
            if late:
                wins, total, d, dlo, dhi = pooled(late)
                print(f"  from {args.start} games : {len(late)} checkpoints, pooled {wins}/{total} = "
                      f"{wins / total:.3f} -> {d:+.0f} Elo (95% CI {dlo:+.0f} .. {dhi:+.0f})")
            else:
                print(f"  from {args.start} games : no checkpoints")
        for games, m1, m2, w, n in shown:
            print(f"  {games:>7}  {w:>4}/{n:<4}  {elo(w / n):+6.0f}")

    ax.axhline(0.0, color=MUTED, linewidth=1, zorder=2)
    if args.start is not None:
        ax.axvline(args.start, color=MUTED, linewidth=1, linestyle=":", zorder=2)
    ax.set_xlabel("self-play games", fontsize=15, color=MUTED, fontweight="bold")
    ax.set_ylabel("Elo difference", fontsize=15, color=MUTED, fontweight="bold")
    ax.tick_params(labelsize=15, colors=MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside : never covers data
    ax.margins(x=0.08)

    fig.tight_layout()
    out = args.out or args.logs[0].with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
