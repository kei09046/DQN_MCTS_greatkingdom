"""Plot losses, self-play statistics and Elo over time from a training log (stdout of `play train ...`).

usage: python plot_training.py train_result_J.txt [-o out.png] [--show] [--batch 512] [--steps 2]
"""
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

# reference categorical order (blue, orange, aqua), muted ink for annotations
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LOSS_ROWS = {0: "policy", 1: "value", 2: "score", 4: "score_map", 5: "entropy"}


def mean(values):
    return sum(values) / len(values) if values else None


def parse_log(path):
    """Returns (saves, comparisons). One save block per 'train_iter' line, in log order."""
    lines = Path(path).read_text(errors="replace").splitlines()
    saves, comparisons = [], []
    cur, last_win_count, games_saved = None, None, None

    for i, line in enumerate(lines):
        if m := re.match(r"model properly saved (\d+)", line):
            games_saved = int(m.group(1))
        elif m := re.match(r"train_iter : (\d+)", line):
            cur = {"train_iter": int(m.group(1)), "games": games_saved}
            saves.append(cur)
        elif cur is None:
            continue
        elif m := re.match(r"games played : ([\d.]+)", line):
            cur["segments"] = float(m.group(1))
        elif m := re.match(r"capture ratio : ([\d.eE+-]+)", line):
            cur["capture_ratio"] = float(m.group(1))
        elif m := re.match(r"average game length : ([\d.eE+-]+)", line):
            cur["segment_length"] = float(m.group(1))
        elif m := re.match(r"replay buffer : (\d+) stored, sampling window (\d+), (\d+) positions added", line):
            cur["stored"], cur["window"], cur["added"] = map(int, m.groups())
        elif line.startswith("train losses"):
            for row, name in LOSS_ROWS.items():
                if i + 1 + row < len(lines):
                    try:
                        values = [float(x) for x in lines[i + 1 + row].split()]
                    except ValueError:
                        values = []
                    cur[name] = mean(values)
        elif m := re.match(r"win count : (\d+)", line):
            last_win_count = int(m.group(1))
        elif m := re.match(r"model (\S+) vs (\S+) winrate ([\d.eE+-]+)", line):
            comparisons.append({"model": m.group(1), "opponent": m.group(2), "winrate": float(m.group(3)),
                                "wins": last_win_count, "games": games_saved, "fallback": False})
        elif line.startswith("model fallback") and comparisons:
            comparisons[-1]["fallback"] = True

    return saves, comparisons


def add_derived(saves, batch, steps):
    """samples trained per new position, and KL = policy loss - target entropy."""
    prev_iter, prev_added = 0, 0
    for s in saves:
        positions = s["added"] - prev_added if "added" in s else s.get("segments", 0) * s.get("segment_length", 0)
        trained = (s["train_iter"] - prev_iter) * batch * steps
        s["samples_per_position"] = trained / positions if positions else None
        prev_iter, prev_added = s["train_iter"], s.get("added", prev_added)
        if s.get("policy") is not None and s.get("entropy") is not None:
            s["kl"] = s["policy"] - s["entropy"]


def elo_chain(comparisons, default_games):
    """Elo of each compared model, chained through its opponent. A model compared against an untrained net
    ('none') becomes the anchor at 0, so its very uncertain win rate against random play doesn't widen every
    later error bar. Win rates of 0 / 1 are shrunk by +0.5 / +1 so they stay finite. Returns list of dicts."""
    rating = {}
    out = []
    for c in comparisons:
        n = round(c["wins"] / c["winrate"]) if c["wins"] and c["winrate"] > 0 else default_games
        wins = c["wins"] if c["wins"] is not None else c["winrate"] * n
        p = (wins + 0.5) / (n + 1)
        elo = 400 * math.log10(p / (1 - p))
        se = 400 / math.log(10) * math.sqrt(p * (1 - p) / n) / (p * (1 - p))
        if c["opponent"] == "none":
            rating[c["model"]] = (0.0, 0.0)
        else:
            base, base_se = rating.get(c["opponent"], (0.0, 0.0))
            rating[c["model"]] = (base + elo, math.hypot(base_se, se))
        games = int(re.search(r"(\d+)", c["model"]).group(1)) if re.search(r"(\d+)", c["model"]) else None
        out.append({**c, "games_model": games, "elo": rating[c["model"]][0], "se": rating[c["model"]][1],
                    "diff": elo, "diff_se": se, "n": n})
    return out


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def series(ax, saves, key, color, label=None):
    pts = [(s["games"], s[key]) for s in saves if s.get(key) is not None]
    if pts:
        x, y = zip(*pts)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3, label=label)


def plot(saves, ratings, out, show):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    ax = axes.flat

    series(ax[0], saves, "policy", BLUE, "policy loss")
    series(ax[0], saves, "entropy", ORANGE, "target entropy")
    series(ax[0], saves, "kl", AQUA, "KL (loss - entropy)")
    style(ax[0], "Policy", "nats")
    if any(s.get("entropy") is not None for s in saves):
        ax[0].legend(fontsize=8, frameon=False)

    for a, key, title in ((ax[1], "value", "Value loss"), (ax[2], "score", "Score loss"), (ax[3], "score_map", "Score map loss")):
        series(a, saves, key, BLUE)
        style(a, title, "loss")

    # fallbacks reload the best model (and reset the optimizer): mark them on the loss panels
    for r in ratings:
        if r["fallback"]:
            for a in ax[:4]:
                a.axvline(r["games_model"], color=MUTED, linestyle="--", linewidth=1)
    ax[1].annotate("dashed = fallback", (0.98, 0.95), xycoords="axes fraction", ha="right", fontsize=7, color=MUTED)

    # Elo, chained through the gating comparisons
    anchor = next((r["model"] for r in ratings if r["opponent"] == "none"), None)
    if ratings:
        x = [r["games_model"] for r in ratings]
        ax[4].errorbar(x, [r["elo"] for r in ratings], yerr=[1.96 * r["se"] for r in ratings], color=BLUE,
                       linewidth=2, marker="o", markersize=4, capsize=3, elinewidth=1)
        for r in ratings:
            if r["fallback"]:
                ax[4].annotate("fallback", (r["games_model"], r["elo"]), textcoords="offset points", xytext=(0, -14),
                               ha="center", fontsize=7, color=MUTED)
    style(ax[4], "Elo (chained gating results, 95% CI)", f"Elo vs {anchor.replace('.pt', '')}" if anchor else "Elo")

    # head-to-head result of each comparison
    if ratings:
        x = [r["games_model"] for r in ratings]
        ax[5].bar(x, [r["diff"] for r in ratings], width=(x[1] - x[0]) * 0.6 if len(x) > 1 else 50, color=BLUE)
        ax[5].errorbar(x, [r["diff"] for r in ratings], yerr=[1.96 * r["diff_se"] for r in ratings], fmt="none",
                       ecolor=MUTED, capsize=3, elinewidth=1)
        ax[5].axhline(0, color=MUTED, linewidth=1)
        for r, xi in zip(ratings, x):
            ax[5].annotate(r["opponent"].replace(".pt", ""), (xi, 0), textcoords="offset points",
                           xytext=(0, 3 if r["diff"] < 0 else -10), ha="center", fontsize=6, color=MUTED)
    style(ax[5], "Each checkpoint vs current best (label = opponent)", "Elo difference")

    series(ax[6], saves, "capture_ratio", BLUE)
    style(ax[6], "Capture ratio (share of segments ending in capture)", "ratio")
    series(ax[7], saves, "segment_length", BLUE)
    style(ax[7], "Average segment length", "moves")
    series(ax[8], saves, "samples_per_position", BLUE)
    style(ax[8], "Training samples per new position", "samples / position")

    for a in axes.flat:
        a.set_xlabel("self-play games", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    if show:
        plt.show()


def print_table(saves, ratings):
    cols = ["games", "train_iter", "samples_per_position", "capture_ratio", "segment_length",
            "policy", "entropy", "kl", "value", "score", "score_map", "window"]
    print(" ".join(f"{c[:10]:>10}" for c in cols))
    for s in saves:
        print(" ".join(f"{s[c]:>10.3f}" if isinstance(s.get(c), float) else f"{str(s.get(c, '-')):>10}" for c in cols))
    print()
    for r in ratings:
        print(f"{r['model']:>12} vs {r['opponent']:<12} winrate {r['winrate']:.3f} ({r['n']} games)  "
              f"diff {r['diff']:+7.1f} ± {1.96 * r['diff_se']:5.1f}  chained Elo {r['elo']:7.1f}"
              + ("  fallback" if r["fallback"] else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("-o", "--out", help="output image (default: <log>.png)")
    parser.add_argument("--show", action="store_true", help="also open a window")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--steps", type=int, default=2, help="optimizer steps per train_iter")
    parser.add_argument("--games", type=int, default=96, help="games per comparison if the log doesn't say")
    args = parser.parse_args()

    saves, comparisons = parse_log(args.log)
    add_derived(saves, args.batch, args.steps)
    ratings = elo_chain(comparisons, args.games)
    print_table(saves, ratings)
    plot(saves, ratings, args.out or str(Path(args.log).with_suffix(".png")), args.show)


if __name__ == "__main__":
    main()
