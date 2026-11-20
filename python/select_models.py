"""Pick a subset of checkpoints for an offline rating comparison, from training logs (stdout of `play train ...`).

Every checkpoint in training plays the current best model ("model A vs B winrate W", preceded by "win count : N").
Taken alone, those matches overrate promoted models : a model is promoted exactly when its one 96-game match came
out above 50%, and non-promoted challengers play a single match each, so they constrain nothing. Summing the
promotion margins therefore climbs far too fast.

So the fit (Bradley-Terry, MAP) also assumes that consecutive checkpoints of the training network have similar
ratings : a Gaussian random-walk prior with --step-sigma Elo per checkpoint. A checkpoint's parent is the previous
checkpoint, or the best model after a "model fallback!" (training reloaded the best weights). With that prior,
challengers that lost pull the promoted models next to them back down.

Checkpoints are then walked in training order and one is selected when
  - its fitted rating is at least +X above the last selected model, or
  - at least Y self-play games have passed since the last selected model.
The first and the last candidates are always selected.

usage: python select_models.py results/train_result_M4.txt [more logs of the same run ...]
           [-x 100] [-y 100000] [--all] [--model-dir ../models] [--games 96] [--prior 1.0] [--step-sigma auto]
           [--table ratings.tsv]

The last output line is the selected model list, space separated, ready for `play evaluate_multi`.
Ratings are in Elo, relative to the first node (the untrained "none" model when the log starts from scratch).
They come from on-train comparisons (on_train_compare_config, temp 0.5): use them to choose models, not as final ratings.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

COMPARE_RE = re.compile(r"^model (\S+) vs (\S+) winrate (\S+)")
WIN_COUNT_RE = re.compile(r"^win count : (\d+)")
PROMOTE_RE = re.compile(r"^Best model updated! \S+ to (\S+)")
FALLBACK_RE = re.compile(r"^model fallback!")
GAMES_RE = re.compile(r"_(\d+)B?\.pt$")

ELO = 400.0 / math.log(10.0)  # natural-log strength -> Elo


def canonical(name):
    """M_12b_960B.pt and M_12b_960.pt hold the same weights : one node."""
    return re.sub(r"B\.pt$", ".pt", name)


def best_name(name):
    return re.sub(r"\.pt$", "B.pt", name)


def games_of(name):
    m = GAMES_RE.search(name)
    return int(m.group(1)) if m else -1  # "none" (untrained model) comes first


def parse_logs(paths, games_per_match):
    """Returns (matches, promoted, lineage). matches : list of (a, b, wins of a, games).
    lineage : list of (checkpoint, parent) : the weights each checkpoint was trained from."""
    matches, promoted, lineage = [], set(), []
    parent = None  # what the next checkpoint was trained from
    for path in paths:
        win_count = None
        for line in Path(path).read_text(errors="replace").splitlines():
            if m := WIN_COUNT_RE.match(line):
                win_count = int(m.group(1))
            elif m := COMPARE_RE.match(line):
                a, b, winrate = canonical(m.group(1)), canonical(m.group(2)), float(m.group(3))
                wins = win_count if win_count is not None else round(winrate * games_per_match)
                if abs(wins / games_per_match - winrate) > 1e-3:
                    print(f"warning : {a} vs {b} : win count {wins} / {games_per_match} games does not match winrate "
                          f"{winrate}; check --games", file=sys.stderr)
                matches.append((a, b, wins, games_per_match))
                win_count = None
                # the first checkpoint of a run is trained from its first opponent (the untrained "none" model).
                lineage.append((a, parent if parent is not None else b))
                parent, best = a, b
            elif m := PROMOTE_RE.match(line):
                promoted.add(canonical(m.group(1)))
                best = canonical(m.group(1))
            elif FALLBACK_RE.match(line):
                parent = best  # training reloaded the best model's weights
    return matches, promoted, lineage


def fit_bradley_terry(matches, lineage, prior, step_sigma):
    """Newton's method on the Bradley-Terry log-likelihood plus a Gaussian random-walk prior (step_sigma Elo) along
    the lineage edges. Every pair gets `prior` virtual wins for each side, so 0% / 100% results stay finite.
    The first node (fewest games) is fixed at 0. Returns (names, ratings in Elo, posterior standard errors in Elo)."""
    names = sorted({n for a, b, _, _ in matches for n in (a, b)}, key=lambda n: (games_of(n), n))
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    wins = np.zeros((n, n))  # wins[i, j] : games i won against j
    for a, b, w, g in matches:
        wins[idx[a], idx[b]] += w
        wins[idx[b], idx[a]] += g - w
    played = wins + wins.T
    wins += prior * (played > 0)
    played = wins + wins.T

    # random-walk prior : graph Laplacian of the lineage edges, scaled by 1 / sigma^2 (natural-log units).
    lap = np.zeros((n, n))
    if step_sigma is not None:
        k = (ELO / step_sigma) ** 2
        for child, par in lineage:
            if child != par:
                i, j = idx[child], idx[par]
                lap[i, i] += k
                lap[j, j] += k
                lap[i, j] -= k
                lap[j, i] -= k

    if n > 1:
        _check_connected(names, played + np.abs(lap))

    s = np.zeros(n)  # natural-log strengths
    free = slice(1, n)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(s[None, :] - s[:, None]))  # p[i, j] : P(i beats j)
        grad = wins.sum(axis=1) - (played * p).sum(axis=1) - lap @ s
        w = played * p * (1.0 - p)
        hess = np.diag(w.sum(axis=1)) - w + lap  # negative Hessian
        step = np.linalg.solve(hess[free, free], grad[free])
        s[free] += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        print("warning : Bradley-Terry fit did not converge", file=sys.stderr)

    cov = np.linalg.inv(hess[free, free])
    se = np.concatenate([[0.0], np.sqrt(np.diag(cov))])

    # Laplace approximation of log p(results | step_sigma), used to pick step_sigma.
    p = 1.0 / (1.0 + np.exp(s[None, :] - s[:, None]))
    log_lik = np.sum(wins * np.log(np.where(played > 0, p, 1.0)))
    evidence = log_lik - 0.5 * s @ lap @ s - 0.5 * np.linalg.slogdet(hess[free, free])[1]
    if step_sigma is not None:
        evidence += 0.5 * np.linalg.slogdet(lap[free, free])[1]
    return names, s * ELO, se * ELO, evidence


def fit_auto_sigma(matches, lineage, prior):
    """Picks step_sigma (Elo per checkpoint) maximizing the Laplace-approximated evidence over a log grid."""
    best = None
    for sigma in np.exp(np.linspace(np.log(2.0), np.log(200.0), 41)):
        fit = fit_bradley_terry(matches, lineage, prior, sigma)
        if best is None or fit[3] > best[1][3]:
            best = (sigma, fit)
    return best


def _check_connected(names, played):
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(played[i])[0]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    if len(seen) != len(names):
        missing = [names[i] for i in range(len(names)) if i not in seen]
        sys.exit(f"comparison graph is not connected (pass every log of the run). unreachable : {missing[:5]} ...")


def select(candidates, rating, min_gain, max_gap):
    """candidates : names in training order. Returns the selected names."""
    if not candidates:
        return []
    selected = [candidates[0]]
    for name in candidates[1:]:
        last = selected[-1]
        if rating[name] - rating[last] >= min_gain or \
                (max_gap is not None and games_of(name) - games_of(last) >= max_gap):
            selected.append(name)
    if selected[-1] != candidates[-1]:
        selected.append(candidates[-1])
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="training logs of one run (all of them, if the run was resumed)")
    ap.add_argument("-x", "--min-gain", type=float, default=100.0,
                    help="select a model once it is this many Elo above the last selected one (default 100)")
    ap.add_argument("-y", "--max-gap", type=int, default=None,
                    help="also select a model once this many self-play games passed since the last selected one")
    ap.add_argument("--all", action="store_true",
                    help="consider every checkpoint, not only promoted (best) models. Non-promoted checkpoints "
                         "have a single 96-game comparison each, so their ratings are noisy")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="only consider models whose file exists in this directory")
    ap.add_argument("--games", type=int, default=96, help="games per on-train comparison (compare_game_cnt)")
    ap.add_argument("--prior", type=float, default=1.0, help="virtual wins given to each side of every pair")
    ap.add_argument("--step-sigma", default="auto",
                    help="random-walk prior : expected Elo change between consecutive checkpoints. 'auto' (default) "
                         "picks the value the results support best; 0 disables the prior (raw match fit, overrates)")
    ap.add_argument("--table", type=Path, default=None, help="write every node's fitted rating to this TSV file")
    args = ap.parse_args()

    matches, promoted, lineage = parse_logs(args.logs, args.games)
    if not matches:
        sys.exit("no 'model A vs B winrate W' lines found")
    if args.step_sigma == "auto":
        sigma, (names, ratings, ses, _) = fit_auto_sigma(matches, lineage, args.prior)
        print(f"step sigma (auto) : {sigma:.1f} Elo per checkpoint")
    else:
        sigma = float(args.step_sigma) if float(args.step_sigma) > 0 else None
        names, ratings, ses, _ = fit_bradley_terry(matches, lineage, args.prior, sigma)
    rating = dict(zip(names, ratings))

    def file_name(name):  # promoted models are kept as ...B.pt
        return best_name(name) if name in promoted else name

    def available(name):
        if args.model_dir is None:
            return True
        return (args.model_dir / file_name(name)).exists() or (args.model_dir / name).exists()

    def on_disk_name(name):
        if args.model_dir is not None and not (args.model_dir / file_name(name)).exists():
            return name
        return file_name(name)

    candidates = [n for n in names if games_of(n) >= 0 and (args.all or n in promoted) and available(n)]
    selected = select(candidates, rating, args.min_gain, args.max_gap)

    if args.table is not None:
        with open(args.table, "w") as f:
            f.write("model\tgames\trating\tse\tpromoted\tselected\n")
            chosen = set(selected)
            for n, r, se in zip(names, ratings, ses):
                f.write(f"{n}\t{games_of(n)}\t{r:.1f}\t{se:.1f}\t{int(n in promoted)}\t{int(n in chosen)}\n")

    print(f"{len(matches)} comparisons, {len(names)} models, {len(promoted)} promoted, {len(candidates)} candidates")
    print(f"{'model':<28}{'games':>9}{'rating':>9}{'+-se':>7}{'gain':>8}")
    prev = None
    for n in selected:
        gain = "" if prev is None else f"{rating[n] - rating[prev]:+.0f}"
        print(f"{on_disk_name(n):<28}{games_of(n):>9}{rating[n]:>9.0f}{ses[names.index(n)]:>7.0f}{gain:>8}")
        prev = n
    print(f"{len(selected)} selected")
    print(" ".join(on_disk_name(n) for n in selected))


if __name__ == "__main__":
    main()
