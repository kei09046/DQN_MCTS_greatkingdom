import numpy as np
import copy


class RuleManager:

    Max = 2 ** 16 - 1
    boardSize = 9
    neutral = [(4, 4)]
    penalty = 2.5
    x_adj = [-1, 0, 1, 0]
    y_adj = [0, 1, 0, -1]

    def __init__(self, board_size=None, neutral=None, penalty=None):
        self.turn = 1
        self.pp = False
        self.territory = [0, 0]
        if board_size is not None:
            RuleManager.boardSize = board_size
        if neutral is not None:
            RuleManager.neutral = neutral
        if penalty is not None:
            RuleManager.penalty = penalty

        # 플레이어가 보는 판
        self.board = np.pad(np.zeros((self.boardSize, self.boardSize), dtype=int),
                            ((1, 1), (1, 1)), 'constant', constant_values=(RuleManager.Max, RuleManager.Max))
        # 돌의 사활 결정
        self.st_board = np.pad(np.zeros((self.boardSize, self.boardSize), dtype=int), ((1, 1), (1, 1)),
                               'constant', constant_values=(RuleManager.Max, RuleManager.Max))
        self.liberty_list = np.zeros(self.boardSize * self.boardSize, dtype=int)
        self.st_cnt = 0
        # 집 결정
        self.bound_board = np.pad(np.zeros((self.boardSize, self.boardSize), dtype=int),
                                 ((1, 1), (1, 1)), 'constant', constant_values=(-1, -1))

        # 1이면 흑집 -1이면 백집
        # 다른 판과 크기 맞춰주기 위해 +2
        self.terr_board = np.zeros((self.boardSize + 2, self.boardSize + 2), dtype=int)
        self.area_cnt = 0
        # 영역 계산 알고리즘(calc)에 필요
        self.eq_list = np.empty(self.boardSize * self.boardSize + 2, dtype=int)
        # 0 : 흑 돌과 인접한 칸이면 true 아니면 false
        self.cl = np.zeros((2, self.boardSize + 2, self.boardSize + 2), dtype=bool)
        # 학습 시 컴퓨터가 착수할 수 있는 영역 -1:pass
        self.available = list(range(self.boardSize * self.boardSize))
        self.available.append(-1)

        for i in RuleManager.neutral:
            self.available.remove(RuleManager.convert(i, is_pair=True))
        self.seq = []

        for cord in RuleManager.neutral:
            self.board[cord[0] + 1][cord[1] + 1] = RuleManager.Max
            self.st_board[cord[0] + 1][cord[1] + 1] = RuleManager.Max
            self.bound_board[cord[0] + 1][cord[1] + 1] = -1

    def current_state(self):
        ret = np.zeros((4, self.boardSize, self.boardSize), dtype=float)
        s = self.board[1:-1, 1:-1].astype(float)
        ret[1][s * self.turn < 0] = 1.0
        ret[0][s * self.turn > 0] = 1.0
        for n in self.neutral:
            ret[1][n[0]][n[1]] = 1.0
            ret[0][n[0]][n[1]] = 1.0
        if self.seq:
            last_move = self.seq[-1]
            ret[2][last_move[0]][last_move[1]] = 1.0
        ret[3][:, :] = self.turn
        return ret

    def available_move(self):
        return self.available

    # 흑 승: 1반환, 무승부: 0반환, 백 승: -1반환
    def end_game(self):
        b_score, w_score = self.score()
        diff = b_score - w_score - self.penalty
        if diff > 0:
            return 1, diff
        if diff == 0:
            return 0, 0
        return -1, -diff

    # 착수 시 승리 : 1반환, 패배 : -1반환, 지속 : 0반환 계가 : -2반환
    def make_move(self, x, y=None):
        if (x == -1 or y == -1) and self.pp:
            self.seq.append((x, y))
            return -2
        if x == -1 or y == -1:
            self.seq.append((x, y))
            self.pp = True
            self.turn *= -1
            return 0
        if y is None:
            pair = RuleManager.convert(x, is_pair=False)
            return self.make_move(pair[0], pair[1])
        self.seq.append((x, y))

        self.available.remove(RuleManager.convert((x, y), is_pair=True))

        # 패딩된 행렬 연산 위해서 +1
        x += 1
        y += 1

        # 돌이 있는 곳에 착수했는지 체크
        if not (self.st_board[x][y] == 0):
            return 0

        # 상대 집에 착수했는지 체크
        if self.terr_board[x][y] * self.turn == -1:
            return -1

        self.pp = False

        # 돌의 활로를 계산, 잡힌 돌이 있는지 판단
        self.board[x][y] = self.turn
        empty_space = 0
        same_adj = set()
        diff_adj = set()

        for i in range(4):
            u = self.board[x + self.x_adj[i]][y + self.y_adj[i]]
            if u == 0:
                empty_space += 1
            elif u == RuleManager.Max:
                pass
            elif u * self.turn > 0:
                same_adj.add(self.st_board[x + self.x_adj[i]][y + self.y_adj[i]])
            else:
                diff_adj.add(self.st_board[x + self.x_adj[i]][y + self.y_adj[i]])

        # print(empty_space)
        s = len(same_adj)
        d = len(diff_adj)

        # 인접한 상대 돌의 활로 감소
        for i in range(d):
            temp = diff_adj.pop()
            self.liberty_list[temp] -= 1
            if self.liberty_list[temp] == 0:
                return 1

        # 인접한 내 돌이 없는 경우
        if s == 0:
            self.st_cnt += 1
            self.st_board[x][y] = self.st_cnt
            self.liberty_list[self.st_cnt] = empty_space
            m = self.st_cnt

        # 인접한 내 돌이 있는 경우
        else:
            m = min(same_adj)
            same_adj.discard(m)
            self.st_board[x][y] = m

            for i in range(self.boardSize):
                for j in range(self.boardSize):
                    if self.st_board[i + 1][j + 1] in same_adj:
                        self.st_board[i + 1][j + 1] = m

            for i in range(s - 1):
                temp = same_adj.pop()
                self.liberty_list[m] += self.liberty_list[temp]
                self.liberty_list[temp] = 0

            self.liberty_list[m] += empty_space - s

        if self.liberty_list[m] == 0:
            return -1

        # print(self.liberty_list)

        # 영역 갱신
        self.bound_board[x][y] = -1
        for i in range(4):
            self.cl[int((1 - self.turn) / 2)][x + self.x_adj[i]][y + self.y_adj[i]] = True

        # 자기 집 안에 착수한 경우
        if self.terr_board[x][y] == self.turn:
            self.territory[int((-self.turn + 1)/2)] -= 1
            self.terr_board[x][y] = 0

        # 공배에 착수한 경우
        else:
            for i in range(4):
                nx = x + self.x_adj[i]
                ny = y + self.y_adj[i]
                if self.bound_board[nx][ny] != -1:
                    self.calc(self.bound_board[nx][ny])
                    break

        # print(self.territory)
        self.turn *= -1
        return 0

    def score(self):
        return self.territory[0], self.territory[1]

    def calc(self, where):
        if where < 0:
            return
        if where > self.area_cnt:
            print("error")

        c = copy.deepcopy(self.bound_board)
        # print(where)
        cnt = 0
        adj = np.full((4, 6), False, dtype=bool)

        for i in range(len(self.eq_list)):
            self.eq_list[i] = i

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if c[i + 1][j + 1] == where:
                    if c[i][j + 1] == -1 and c[i + 1][j] == -1:
                        c[i + 1][j + 1] = cnt
                        cnt += 1

                    elif c[i][j + 1] == -1:
                        c[i + 1][j + 1] = c[i + 1][j]

                    elif c[i + 1][j] == -1:
                        c[i + 1][j + 1] = c[i][j + 1]

                    elif c[i][j + 1] == c[i + 1][j]:
                        c[i + 1][j + 1] = c[i][j + 1]

                    elif c[i][j + 1] < c[i + 1][j]:
                        c[i + 1][j + 1] = c[i][j + 1]
                        self.__modify(c[i + 1][j], c[i][j + 1])

                    else:
                        c[i + 1][j + 1] = c[i + 1][j]
                        self.__modify(c[i][j + 1], c[i + 1][j])

        dnt = 0
        for i in range(cnt):
            if i == self.eq_list[i]:
                self.eq_list[i] = dnt
                dnt += 1
            else:
                self.eq_list[i] = self.eq_list[self.eq_list[i]]

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.bound_board[i + 1][j + 1] > where:
                    self.bound_board[i + 1][j + 1] -= 1
                elif self.bound_board[i + 1][j + 1] == where:
                    self.bound_board[i + 1][j + 1] = self.area_cnt + self.eq_list[c[i + 1][j + 1]]

        if dnt > 4:
            print("dnt to big")
            print(self.board, "move sequence ", self.seq)

        # print(self.bound_board)

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.bound_board[i + 1][j + 1] >= self.area_cnt:
                    d = self.bound_board[i + 1][j + 1] - self.area_cnt
                    # print(d)
                    if i == 0:
                        adj[d][0] = True
                    if j == 0:
                        adj[d][1] = True
                    if i == self.boardSize - 1:
                        adj[d][2] = True
                    if j == self.boardSize - 1:
                        adj[d][3] = True
                    if self.cl[0][i + 1][j + 1]:
                        adj[d][4] = True
                    if self.cl[1][i + 1][j + 1]:
                        adj[d][5] = True

        # print(cnt, dnt)
        t = [[False for i in range(2)] for j in range(4)]
        for i in range(dnt):
            t[i][0] = not (adj[i][0] and adj[i][1] and adj[i][2] and adj[i][3]) and not adj[i][5]
            t[i][1] = not (adj[i][0] and adj[i][1] and adj[i][2] and adj[i][3]) and not adj[i][4]

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                u = self.bound_board[i + 1][j + 1] - self.area_cnt
                if u >= 0 and t[u][0]:
                    self.terr_board[i + 1][j + 1] = 1
                    self.available.remove(RuleManager.convert((i, j), is_pair=True))
                    self.territory[0] += 1
                if u >= 0 and t[u][1]:
                    self.terr_board[i + 1][j + 1] = -1
                    self.available.remove(RuleManager.convert((i, j), is_pair=True))
                    self.territory[1] += 1

        self.area_cnt += dnt - 1
        # print(self.bound_board)
        return

    def __modify(self, n, m):
        eqv = self.eq_list[n]
        if eqv == n:
            self.eq_list[n] = m
            return
        if eqv > n:
            self.eq_list[n] = m
            self.__modify(eqv, m)
            return
        self.eq_list[n] = min(eqv, m)
        self.__modify(max(eqv, m), min(eqv, m))
        return

    @staticmethod
    def convert(no, is_pair=False):
        if is_pair:
            return no[0] * RuleManager.boardSize + no[1]
        else:
            return no // RuleManager.boardSize, no % RuleManager.boardSize


import os
import re
import subprocess
import threading
import queue
import math
import time
import random
import json
from datetime import datetime

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# 실제 게임이 실행되는 파일 GUI 담당이기도 함

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
MATCHES_DIR = os.path.join(REPO_ROOT, "matches")

BOARD_BG = "#DCB35C"
OVERLAY_COLOR = "#1565C0"  # blended toward this color for stronger policy/visit intensity
# A proven win/loss (root->forcedState != 0, see Analysis::printAnalysis in src/analysis.cpp) leaves
# exactly one candidate in "moves" -- the winning move (forced > 0) or the best delaying move
# (forced < 0). Flat, unblended colors (not run through _blend_hex like the ordinary
# visit-share gradient) so a proven result reads as categorically different from "just happens
# to currently hold 100% of the visits" rather than another shade of the same gradient.
FORCED_WIN_COLOR = "#1b8a3a"
FORCED_LOSS_COLOR = "#b23b3b"

# A Node only gets its own per-move breakdown (edgeP/edgeN -- see Analysis::printAnalysis in
# src/analysis.cpp) once *something* has searched through it as an internal node, not merely
# visited it once as a leaf to evaluate it; a freshly jumped-to child can need up to 2 fresh
# playouts to reach that point (1 to evaluate it, 1 more to expand it). Requesting this many
# after a paused move is enough to make its own children's real prior/stats visible instead of
# a blank all-zero breakdown, while still being a no-op (see the "analyze" protocol's
# cumulative-target semantics) for any child that already had more visits than this from
# before -- so it never grows into a real "resume search".
_PAUSED_MOVE_MIN_VISITS = 2


def _blend_hex(c1, c2, t):
    """Lerp between two '#rrggbb' colors by t in [0, 1]; used to fake per-marker opacity
    (Tk canvas fills have no real alpha) by blending toward the board background instead."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _luminance(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _policy_visit_cross_entropy(result):
    """H(visit distribution, policy prior) -- same quantity AlphaZero-style training uses as
    the policy head's loss, with the search's own visit counts standing in for the target."""
    total_visits = result.get("visits") or 0
    moves = result.get("moves", [])
    if total_visits <= 0 or not moves:
        return None

    eps = 1e-8
    ce = 0.0
    for mv in moves:
        visits = mv.get("visits", 0)
        if visits <= 0:
            continue
        target = visits / total_visits
        pred = max(mv.get("prior", 0.0), eps)
        ce -= target * math.log(pred)
    return ce


def _value_mse(result):
    """Squared error between the network's first-look value estimate (initQ, before any search)
    and the fully-searched winrate -- how far search moved the evaluation from its first guess."""
    init_q = result.get("initQ")
    winrate = result.get("winrate")
    if init_q is None or winrate is None:
        return None
    return (init_q - winrate) ** 2


# Matches configs/play_config.json's mcts.cPuct -- the config "./play analyze" itself loads (see
# ModelCompare::analyze / main.cpp's `mod == "analyze"` branch) -- and is the same constant
# Node::selectChildInSearch (src/PMCTS.cpp) uses for its PUCT term below.
CPUCT = 1.5


def _move_preference(mv, total_visits):
    """Node::selectChildInSearch's PUCT preference value (src/PMCTS.cpp:261-281) for an
    already-visited child -- the quantity search itself ranks moves by when picking what to
    explore next on the following playout. Every move ever shown in the candidate list has
    visits > 0 (see the filter in _sorted_candidate_moves), i.e. an already-expanded child, so
    the fpu/unvisited-child branch never applies here -- only these three, picked by the child's
    own "forced" field (src/PMCTS.cpp:266-281, see analysis.cpp's childForced comment for the
    sign convention -- negative there means THIS move is a proven win for root's own mover):

    - forced < 0 (proven win): selectChildInSearch returns this child immediately, ahead of any
      pref comparison at all -- not merely "highest pref". +inf here means exactly that: it
      always wins any comparison, the same way the real search short-circuits straight to it
      (and, when multiple children are wins, the real engine takes whichever it hits first in
      board order rather than ranking them against each other -- +inf ties do the same: it
      groups them all at the top without claiming to rank among them).
    - forced > 0 (proven loss): pref is suppressed to a fixed -2.0 + ..., not the ordinary
      q-based formula -- almost never selected over any non-losing sibling.
    - forced == 0 (undetermined): the ordinary formula,
          pref = child.W/child.N + cPuct * prior * sqrt(rootVisits) / (1 + child.N)
      child.W/child.N is exactly the "q" field already parsed per move; child.N/edgeN is
      "visits"."""
    forced = mv.get("forced", 0)
    puct_term = CPUCT * mv["prior"] * math.sqrt(max(total_visits, 0)) / (1 + mv["visits"])
    if forced < 0:
        return math.inf
    if forced > 0:
        return -2.0 + puct_term
    return mv["q"] + puct_term


# Field each sortable column reads off a move dict (see _move_preference for "pref").
_CANDIDATE_SORT_KEYS = {
    "visits": lambda mv: mv["visits"],
    "policy": lambda mv: mv["prior"],
    "winrate": lambda mv: mv["winrate"],
    "q": lambda mv: mv["q"],
    "pref": lambda mv: mv["pref"],
}

# (sort key or None if not sortable, header text, field width) for each column of the candidate
# Listbox -- shared by _build_search_column (builds the clickable header row) and
# _update_candidate_listbox (formats each row and refreshes the header's sort-indicator arrow).
_CANDIDATE_COLUMNS = (
    (None, "move", 10),
    ("visits", "visits", 8),
    ("policy", "policy", 8),
    ("winrate", "winrate", 9),
    ("q", "Q", 8),
    ("pref", "pref", 8),
)


def _sorted_candidate_moves(result, sort_key="visits", reverse=True):
    """Candidate moves with at least one visit, sorted by the given field -- the exact set/order
    the candidate Listbox is populated in, so a row index maps back to the same move on either
    side. Every move dict also gets a "pref" field filled in (see _move_preference) regardless of
    whether "pref" is the active sort key, since the Listbox always displays it as a column."""
    total_visits = result.get("visits") or 0
    moves = result.get("moves", [])
    for mv in moves:
        mv["pref"] = _move_preference(mv, total_visits)
    key_fn = _CANDIDATE_SORT_KEYS.get(sort_key, _CANDIDATE_SORT_KEYS["visits"])
    moves = sorted(moves, key=key_fn, reverse=reverse)
    return [mv for mv in moves if mv["visits"] > 0]


_FLOAT = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"

# ---- ModelCompare::analyze() protocol (see Analysis::printAnalysis in src/analysis.cpp) ----
# "winrate : <float>" (-1 = 0%, 1 = 100%, from the perspective of whoever is to move)
_ANALYSIS_WINRATE_RE = re.compile(r"^winrate\s*:\s*" + _FLOAT)
# "visits : <int>" total root visit count
_ANALYSIS_VISITS_RE = re.compile(r"^visits\s*:\s*(\d+)")
# "initQ : <float>" -- the raw value-head output the first time this position was evaluated
_ANALYSIS_INITQ_RE = re.compile(r"^initQ\s*:\s*" + _FLOAT)
# "forced : <int>" -- 0 = undetermined, nonzero = proven win/loss. Once set, the engine's
# analyze loop stops for good well short of the requested playout target; distinguishing this
# from "still climbing toward the target" is what the visits status label uses to avoid getting
# stuck showing "analyzing..." forever once the engine has actually finished.
_ANALYSIS_FORCED_RE = re.compile(r"^forced\s*:\s*(-?\d+)")
# "move <r> <c> visits <N> prior <P> winrate <W> q <Q> forced <F> variation <r1> <c1> ..." --
# one candidate move; "forced" is this CHILD's own forcedState (not the root's -- see the
# _ANALYSIS_FORCED_RE root-level field above), from the child's own mover's perspective: negative
# means the child's mover (the opponent) loses there, i.e. THIS move is a proven win for root's
# own mover; positive is the opposite, a proven loss for root's mover (see childForced's comment
# in src/analysis.cpp and Node::selectChildInSearch, src/PMCTS.cpp:266-281). The variation tail is
# 0 or more coordinate pairs, captured whole and split separately since its length varies per move.
_ANALYSIS_MOVE_RE = re.compile(
    r"^move\s+(-?\d+)\s+(-?\d+)\s+visits\s+" + _FLOAT + r"\s+prior\s+" + _FLOAT
    + r"\s+winrate\s+" + _FLOAT + r"\s+q\s+" + _FLOAT + r"\s+forced\s+(-?\d+)\s+variation(.*)$")
# "scoreSearch : <float>" -- search-refined score-head estimate: same backup mechanism as
# winrate (root->S accumulates a scoreExp contribution from every playout along the tree), so
# this moves as search deepens -- the scoreExp/initQ pair's counterpart to winrate/scoreSearch.
_ANALYSIS_SCORESEARCH_RE = re.compile(r"^scoreSearch\s*:\s*" + _FLOAT)
# "scoreExp : <float>" -- the score head's raw scalar output the first time this position was
# evaluated (mirrors initQ, but for the score head): net (Black - White) territory/capture
# margin, from the perspective of whoever is to move (same turn-relative convention as
# winrate/scoreMap), without komi applied (see the PolicyValueOutput comment in src/consts.h).
_ANALYSIS_SCOREEXP_RE = re.compile(r"^scoreExp\s*:\s*" + _FLOAT)
# "scoreMap : v0 v1 ... v80" -- row-major (r*9+c) per-point territory prediction, tanh output
# in [-1, 1]: -1 = certain Black-owned point, +1 = certain White-owned point (see
# Game::makeMove's end-of-game scoreMap in gamerules.cpp for the matching training label).
_ANALYSIS_SCOREMAP_RE = re.compile(r"^scoreMap\s*:\s*(.*)$")
# "playout forced <F> winp <W> score <S> path <r1> <c1> <r2> <c2> ..." -- one playout's search
# path (root -> leaf, as the moves selected along the way) and leaf NN evaluation, only sent
# while debug mode is on (see Analysis::setDebugMode/printPlayoutDebugLine in
# src/analysis.hpp/.cpp).
# F != 0 means the leaf was a proven win/loss, never evaluated by the net -- winp/score are then
# 0 and meaningless. Streamed live, one line per playout the moment it finishes -- nothing is
# stored on the engine side, so this can arrive interleaved with, and outside of, any "analysis
# begin"/"analysis end" block; AnalysisEngine recognizes it independent of that block state, and
# Game is what accumulates the resulting stream into a persistent list (see Game.playout_log).
_ANALYSIS_PLAYOUT_RE = re.compile(
    r"^playout\s+forced\s+(-?\d+)\s+winp\s+" + _FLOAT + r"\s+score\s+" + _FLOAT + r"\s+path(.*)$")


def list_models(models_dir=MODELS_DIR):
    if not os.path.isdir(models_dir):
        return []
    return sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))


class AnalysisEngine:
    """Wraps `./play analyze <model>` (ModelCompare::analyze in src/modelcompare.cpp).

    This never moves on its own: the caller drives the position with reset()/play()
    and explicitly asks for a winrate/policy/visit-count breakdown of every candidate
    move via request_analysis(). Results stream back asynchronously through a reader
    thread. Play-vs-engine is layered on top of this in Game: the GUI itself watches
    the streamed results and, whenever it's a color the engine has been told to play,
    picks the top candidate move (see Game._auto_play_move) and feeds it back in via
    send_play -- there is no separate "engine plays itself" process/protocol anymore.
    """

    def __init__(self, model, build_dir=BUILD_DIR):
        self.proc = subprocess.Popen(
            ["./play", "analyze", model],
            cwd=build_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # results: dicts {"winrate": float, "visits": int,
        #                 "moves": [{"move": (r,c), "visits": float, "prior": float, "winrate": float,
        #                            "q": float, "forced": int}, ...]}
        self.results = queue.Queue()
        # playouts: individual debug-mode playout dicts {"forced": int, "winp": float,
        # "score": float, "path": [(r,c), ...]}, one per line, streamed live and independent of
        # the "analysis begin"/"analysis end" block cycle -- the engine doesn't accumulate these
        # itself (see printPlayoutDebugLine in src/analysis.cpp), so building up the growing list a
        # debug session looks at is entirely on the Python/GUI side; see Game.playout_log.
        self.playouts = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        in_block = False
        current = None

        for line in self.proc.stdout:
            line = line.rstrip("\n")

            # Checked unconditionally, before the in_block gate below -- unlike every other
            # line here, a "playout ..." line isn't part of any analysis snapshot and can show
            # up interleaved with, or outside of, an "analysis begin"/"analysis end" block.
            m = _ANALYSIS_PLAYOUT_RE.match(line)
            if m:
                forced, winp, score, path_tail = m.groups()
                path_ints = [int(x) for x in path_tail.split()]
                self.playouts.put({
                    "forced": int(forced),
                    "winp": float(winp),
                    "score": float(score),
                    "path": list(zip(path_ints[0::2], path_ints[1::2])),
                })
                continue

            if line == "analysis begin":
                in_block = True
                current = {"winrate": None, "visits": None, "initQ": None, "moves": [],
                           "scoreExp": None, "scoreSearch": None, "scoreMap": None, "forced": 0}
                continue
            if line == "analysis end":
                in_block = False
                if current is not None:
                    self.results.put(current)
                current = None
                continue
            if not in_block:
                continue

            m = _ANALYSIS_MOVE_RE.match(line)
            if m:
                r, c, visits, prior, winrate, q, forced, var_tail = m.groups()
                var_ints = [int(x) for x in var_tail.split()]
                variation = list(zip(var_ints[0::2], var_ints[1::2]))
                current["moves"].append({
                    "move": (int(r), int(c)),
                    "visits": float(visits),
                    "prior": float(prior),
                    "winrate": float(winrate),
                    "q": float(q),
                    "forced": int(forced),
                    "variation": variation,
                })
                continue

            m = _ANALYSIS_WINRATE_RE.match(line)
            if m:
                current["winrate"] = float(m.group(1))
                continue

            m = _ANALYSIS_VISITS_RE.match(line)
            if m:
                current["visits"] = int(m.group(1))
                continue

            m = _ANALYSIS_INITQ_RE.match(line)
            if m:
                current["initQ"] = float(m.group(1))
                continue

            m = _ANALYSIS_FORCED_RE.match(line)
            if m:
                current["forced"] = int(m.group(1))
                continue

            m = _ANALYSIS_SCORESEARCH_RE.match(line)
            if m:
                current["scoreSearch"] = float(m.group(1))
                continue

            m = _ANALYSIS_SCOREEXP_RE.match(line)
            if m:
                current["scoreExp"] = float(m.group(1))
                continue

            m = _ANALYSIS_SCOREMAP_RE.match(line)
            if m:
                current["scoreMap"] = [float(x) for x in m.group(1).split()]
                continue

    def _send(self, cmd):
        if self.proc.poll() is not None or self.proc.stdin.closed:
            return
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def send_reset(self):
        self._send("reset")

    def send_play(self, x, y):
        self._send(f"play {x} {y}")

    def request_analysis(self, playouts=None):
        self._send("analyze" if playouts is None else f"analyze {playouts}")

    def send_pause(self):
        self._send("pause")

    def send_debug(self, on):
        self._send("debug on" if on else "debug off")

    def get_result_nowait(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def get_playout_nowait(self):
        try:
            return self.playouts.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


class MatchRunner:
    """Drives an engine-vs-engine match for the "Engine Match" dialog: two independently
    configured AnalysisEngine processes (each its own `./play analyze <model>` subprocess, so
    each genuinely runs its own model at its own playout budget -- there is no shared engine
    state between them) alternate turns on a position tracked locally by a plain headless
    RuleManager, the same rules implementation used for local play. RuleManager is the sole
    authority on legality/scoring here -- the match doesn't depend on parsing "game over"/score
    text out of either engine process at all, just its own make_move()/end_game() the same way
    Game._commit_move already does for the interactive board. Each engine is only ever asked
    "what would you play here" (via analyze) and then told the resulting move back (send_play,
    to both engines -- including the one that didn't choose it) to keep its own tree in sync.

    Runs entirely on a background thread; progress and finished-game records are handed to the
    caller thread-safely through `events` (a queue.Queue), which the GUI drains with
    root.after() -- same reader-thread-plus-queue pattern AnalysisEngine itself uses.
    """

    def __init__(self, engine_a_cfg, engine_b_cfg, n_games, build_dir=BUILD_DIR):
        # each cfg: {"model": str, "playouts": int, "temp": float}
        self.cfg = {"A": engine_a_cfg, "B": engine_b_cfg}
        self.n_games = n_games
        self.build_dir = build_dir
        self.events = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    @staticmethod
    def _wait_for_analysis(engine, target, stop_event, poll_interval=0.05, timeout=600):
        """Blocks (on the background thread only) until a streamed analysis result reaches the
        requested visit target or the position is proven (forced != 0) -- the same "done"
        condition the GUI's own status label uses, see Game._compute_analysis_status."""
        deadline = time.monotonic() + timeout
        latest = None
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return latest
            r = engine.get_result_nowait()
            if r is not None:
                latest = r
                if r.get("forced") or (r.get("visits") or 0) >= target:
                    # Drain anything else already queued (a chunk or two can land in a burst)
                    # so the next request starts from an empty queue.
                    while engine.get_result_nowait() is not None:
                        pass
                    return latest
            else:
                time.sleep(poll_interval)
        return latest

    @staticmethod
    def _select_move(result, temp, move_count):
        """Same move-selection rule as Node::selectMoveProb in src/PMCTS.cpp (weight ∝
        visits**temp, sampled; forced fully greedy once temp >= 5.0 or move_count >= 10) --
        reimplemented here in Python rather than exposed as a new engine command, since the
        "analyze" protocol already hands over the full visit-count breakdown this needs, and
        every other piece of per-engine configuration in this match feature is likewise owned
        client-side. Note this codebase's temp convention is inverted from the usual: LOW temp
        means MORE random (near-uniform over visited moves), HIGH temp means MORE greedy.
        """
        moves = [mv for mv in result.get("moves", []) if mv["visits"] > 0]
        if not moves:
            moves = result.get("moves", [])
        if not moves:
            return None
        if temp >= 5.0 or move_count >= 10:
            return max(moves, key=lambda mv: mv["visits"])

        weights = [mv["visits"] ** temp if mv["visits"] > 0 else 0.0 for mv in moves]
        total = sum(weights)
        if total <= 0:
            return max(moves, key=lambda mv: mv["visits"])
        pick = random.uniform(0, total)
        upto = 0.0
        for mv, w in zip(moves, weights):
            upto += w
            if upto >= pick:
                return mv
        return moves[-1]

    def _run(self):
        engines = {}
        try:
            engines["A"] = AnalysisEngine(self.cfg["A"]["model"], build_dir=self.build_dir)
            engines["B"] = AnalysisEngine(self.cfg["B"]["model"], build_dir=self.build_dir)

            for game_idx in range(self.n_games):
                if self._stop.is_set():
                    break

                # Alternates colors game to game, same "each engine plays both sides" spirit as
                # ModelCompare::play_match in src/modelcompare.cpp (used for training-time model
                # comparison), just one game per color-assignment instead of a whole half-batch.
                black = "A" if game_idx % 2 == 0 else "B"
                white = "B" if black == "A" else "A"
                self.events.put({"type": "game_start", "index": game_idx, "black": black, "white": white})

                for eng in engines.values():
                    eng.send_reset()

                rm = RuleManager()
                move_log = []
                winner_color, margin, by = None, None, None

                while True:
                    if self._stop.is_set():
                        break
                    # Safety valve: a real game always ends via capture, suicide, or double-pass
                    # scoring well before this -- this only guards against an unforeseen rules
                    # desync leaving the two sides passing the loop back and forth forever.
                    if len(rm.seq) > 4 * RuleManager.boardSize * RuleManager.boardSize:
                        by = "move limit"
                        break

                    mover = black if rm.turn == 1 else white
                    cfg = self.cfg[mover]
                    engines[mover].request_analysis(cfg["playouts"])
                    result = self._wait_for_analysis(engines[mover], cfg["playouts"], self._stop)
                    if self._stop.is_set():
                        break
                    if result is None:
                        by = "aborted (no analysis result)"
                        break

                    chosen = self._select_move(result, cfg["temp"], len(rm.seq))
                    if chosen is None:
                        # Only reachable once forced (see the "per-move breakdown is only
                        # meaningful..." comment in Analysis::printAnalysis): the position is a
                        # proven win/loss but this particular forced child happened to still be
                        # unexpanded (root->child[i] == nullptr) at whatever low visit count
                        # search stopped at, so the engine had no single move to single out
                        # either -- rather than aborting the game with no result, resolve it
                        # directly from forcedState (positive = whoever's on the move here wins,
                        # matching Node::selectMove's own convention).
                        if result.get("forced"):
                            winner_color = rm.turn if result["forced"] > 0 else -rm.turn
                            by = "forced (no move data)"
                        else:
                            by = "aborted (no candidate move)"
                        break

                    r, c = chosen["move"]
                    is_pass = (r, c) == (RuleManager.boardSize, 0)

                    move_log.append({
                        "ply": len(rm.seq),
                        "by": mover,
                        "move": None if is_pass else [r, c],
                        "visits": chosen["visits"],
                        "prior": chosen["prior"],
                        "move_winrate": chosen["winrate"],
                        "move_q": chosen["q"],
                        "root_winrate": result.get("winrate"),
                        "root_visits": result.get("visits"),
                        "root_initQ": result.get("initQ"),
                        "root_scoreExp": result.get("scoreExp"),
                        "root_scoreSearch": result.get("scoreSearch"),
                    })

                    for eng in engines.values():
                        eng.send_play(r, c)

                    if is_pass:
                        code = rm.make_move(-1, 0)
                        if code == -2:
                            winner_color, margin = rm.end_game()
                            by = "score"
                            break
                    else:
                        code = rm.make_move(r, c)
                        if code == 1 or code == -1:
                            # Read immediately: rm.turn hasn't flipped yet for either the
                            # capture-win or self-capture/illegal-territory path (same
                            # `result * rm.turn` reconstruction Game._commit_move uses).
                            winner_color = code * rm.turn
                            by = "capture" if code == 1 else "suicide"
                            break

                winner = None
                if winner_color == 1:
                    winner = black
                elif winner_color == -1:
                    winner = white
                elif winner_color == 0:
                    winner = "draw"
                # else: aborted/move-limit -- winner stays None

                record = {
                    "index": game_idx, "black": black, "white": white,
                    "winner": winner, "winner_color": winner_color,
                    "margin": margin, "by": by, "moves": move_log,
                }
                self.events.put({"type": "game_end", "record": record})

        finally:
            for eng in engines.values():
                eng.close()
            self.events.put({"type": "done"})


class Game:
    def __init__(self, margin=50):
        self.rule = RuleManager()
        self.boardSize = RuleManager.boardSize
        self.neutral = RuleManager.neutral
        self.m = margin
        self.cord = [[(0, 0) for i in range(self.boardSize)] for j in range(self.boardSize)]
        self.delta = int((900 - self.m * 2) / (self.boardSize - 1))
        self.stone_size = int(self.delta * 0.8)
        self.cl = int(self.delta * 0.2)
        self.root = Tk()
        self.root.title("Great Kingdom")
        self.root.geometry("2100x1000")

        self.boardFrame = Frame(self.root)
        self.canvas = Canvas(self.boardFrame, width=900, height=900, bg=BOARD_BG, highlightthickness=0)
        self.passButton = Button(self.boardFrame, text="pass")
        self.statusLabel = Label(self.boardFrame, text="", font=("Helvetica", 14))

        self.sideFrame = Frame(self.root, width=1100, height=900)
        self.sideFrame.pack_propagate(False)
        self.setupPanel = None
        self.analysisResultPanel = None
        # Which field the candidate-move Listbox is currently sorted by, and in which direction --
        # set before the panel is built since _build_search_column's clickable column headers read
        # it to render their initial sort-indicator arrow. See _sorted_candidate_moves/
        # _on_candidate_sort_header_click.
        self.candidate_sort_key = "visits"
        self.candidate_sort_reverse = True
        self._build_setup_panel()
        self._build_analysis_result_panel()

        self.stones = []
        self.stones.append(
            ImageTk.PhotoImage(Image.open("images/b_stone.png").resize((self.stone_size, self.stone_size))))
        self.stones.append(
            ImageTk.PhotoImage(Image.open("images/w_stone.png").resize((self.stone_size, self.stone_size))))
        self.stones.append(
            ImageTk.PhotoImage(Image.open("images/neu_stone.png").resize((self.stone_size, self.stone_size))))
        self.turn = 0
        self.on_stone = [[False for i in range(self.boardSize)] for j in range(self.boardSize)]
        self.game_over = False
        # analysis_mode: a session (AnalysisEngine process) is live for the current position.
        # Which side(s), if any, the engine auto-plays for is tracked independently by
        # engine_plays_black_var/engine_plays_white_var so it can be toggled freely mid-session --
        # unchecking both drops back to pure hands-on analysis without ending the session.
        self.analysis_mode = False
        self.analysisEngine = None
        self.last_analysis_result = None
        self.selected_variation_move = None
        self.analysis_paused = False
        # "analyze N" is a cumulative visit-count target on the engine's side (see
        # modelcompare.cpp), not a one-shot playout budget -- and the tree persists across
        # moves, so a freshly-current position's root visits are often already nonzero (carried
        # over from being explored as a child during the previous move's search). These two
        # track, for whatever position is currently "this turn": analysis_turn_base_visits is
        # that carried-over starting point (visits before any of this turn's own search ran),
        # and analysis_turn_requested is the Playout limit field's value as of the most recent
        # request (_request_more_playouts always recomputes the absolute target as base +
        # this, fresh from the field, rather than piling `extra` on top of however far search
        # has already gotten -- so nudging the field from 4000 to 4001 and hitting Apply runs
        # exactly one more playout, not another 4001 of them). Together they let the Visits
        # label and _compute_analysis_status report "done this turn" instead of the engine's
        # raw, carryover-inflated root visit count. Reset at every turn boundary: see
        # _commit_move (sets analysis_turn_base_visits to the carryover, analysis_turn_requested
        # to the limit) and _sync_analysis_position (both to 0, since a fresh replay carries
        # over nothing).
        self.analysis_turn_base_visits = 0
        self.analysis_turn_requested = 0
        # Guards _maybe_auto_play from re-triggering multiple times for the same position while
        # several analysis result blocks stream in as the search climbs toward the playout limit.
        self._auto_move_seq_len = -1
        # Debug mode: the engine streams individual playout lines the moment each one finishes
        # (see AnalysisEngine.playouts) and stores none of it itself -- playout_log is *this*
        # side's accumulation of that stream (drained in poll_analysis), an ordinary growing
        # list. selected_playout_index indexes into it directly, or is None; only meaningful
        # while debug_mode_var is checked. See _on_debug_mode_change / _draw_playout_path_overlay.
        self.playout_log = []
        self.selected_playout_index = None

        # Every move committed via _commit_move (human click, pass, or engine auto-play) is
        # logged here regardless of session state -- see _commit_move and _save_session_game.
        # Only ever written to matches/ if session_used_engine ends up true for the game (i.e.
        # a live analysis session -- see start_session -- touched it at some point); a pure
        # "New Local Game" leaves this accumulating harmlessly but unsaved. Kept in lock-step
        # with self.rule.seq (same length, same moves) at all times -- see _commit_move's
        # divergence-truncation -- so it doubles as "the move list to browse" in the Moves tab
        # (see _current_moves_list) whenever no recorded game is loaded for reference.
        self.session_move_log = []
        self.session_used_engine = False
        self.session_model = None

        # A previously recorded game (auto-saved session, or a loaded engine-match game -- see
        # _load_game_for_reference) used as the Moves tab's reference data: review_mode locks
        # out board clicks/pass (browsing a fixed, already-finished game, not playing a new one)
        # and review_moves becomes what the Moves tab/position navigation browse instead of
        # session_move_log. review_index counts moves played so far in whichever of the two is
        # current (0 = empty board, len(moves) = final position) -- see _jump_to_ply.
        self.review_mode = False
        self.review_record = None
        self.review_moves = []
        self.review_index = 0

        # Per-ply "Current evaluation" data for the Moves tab (see _update_moves_tab): keyed by
        # ply (== len(self.rule.seq) at the moment a streamed analysis result arrives -- see
        # _update_analysis_result_panel), so re-visiting a former position during live analysis
        # (_jump_to_ply) naturally refreshes that ply's entry instead of only ever recording
        # whatever a move happened to be chosen with the first time through. Cleared whenever a
        # session starts fresh (start_session) -- see its own comment.
        self.position_eval_log = {}

        # The exact ply/status text on_end last produced, for _render_review_position to restore
        # verbatim when a free-play (no game loaded) jump lands back on the position where the
        # actual game ended -- game_over itself doesn't carry the "who won and how" text.
        self.game_over_ply = None
        self.game_over_text = None

    def _build_setup_panel(self):
        panel = Frame(self.sideFrame)
        self.setupPanel = panel

        Label(panel, text="Great Kingdom", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 15))

        Label(panel, text="Start a session", font=("Helvetica", 12, "bold"), anchor="w").pack(fill=X)
        Label(panel, text="Analyze the current position, and optionally have\n"
                          "the engine play one or both sides. You can flip who's\n"
                          "playing which color at any time from the session panel.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(4, 10))

        models = list_models()
        Label(panel, text="Model:", anchor="w").pack(fill=X, pady=(0, 0))
        self.setup_model_var = StringVar(value=models[0] if models else "")
        if models:
            model_row = Frame(panel)
            model_row.pack(fill=X, pady=(0, 10))
            Label(model_row, textvariable=self.setup_model_var, anchor="w",
                  relief=SUNKEN, bg="white", padx=5).pack(side=LEFT, fill=X, expand=True)
            Button(model_row, text="Choose...", command=self._open_model_picker).pack(side=LEFT, padx=(5, 0))
        else:
            Label(panel, text=f"(no .pt files found in {MODELS_DIR})", fg="red",
                  wraplength=260, justify=LEFT, anchor="w").pack(fill=X, pady=(0, 10))

        Label(panel, text="Engine plays:", anchor="w").pack(fill=X)
        self.setup_color_var = StringVar(value="analysis")
        Radiobutton(panel, text="Black (you play White)", variable=self.setup_color_var,
                    value="engine_black", anchor="w").pack(fill=X)
        Radiobutton(panel, text="White (you play Black)", variable=self.setup_color_var,
                    value="engine_white", anchor="w").pack(fill=X)
        Radiobutton(panel, text="Neither -- analysis only", variable=self.setup_color_var,
                    value="analysis", anchor="w").pack(fill=X)

        self.start_session_button = Button(panel, text="Start Session", command=self.start_session)
        self.start_session_button.pack(fill=X, pady=(15, 0))
        if not models:
            self.start_session_button.configure(state=DISABLED)

        Frame(panel, height=1, bg="#999999").pack(fill=X, pady=20)

        Label(panel, text="No engine: the board is free to\nplay locally, move by move.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(0, 10))
        Button(panel, text="New Local Game", command=self.reset_board).pack(fill=X)

        Frame(panel, height=1, bg="#999999").pack(fill=X, pady=20)

        Label(panel, text="Play two independently-configured engines\nagainst each other for a number of games,\n"
                          "recording every move, evaluation, and result.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(0, 10))
        match_button = Button(panel, text="Engine vs Engine Match...", command=self._open_match_dialog)
        match_button.pack(fill=X)
        if not models:
            match_button.configure(state=DISABLED)

        Frame(panel, height=1, bg="#999999").pack(fill=X, pady=20)

        Label(panel, text="Load a previously played game -- an\n"
                          "auto-saved human/engine game, or one game\n"
                          "out of a saved engine match -- into the\n"
                          "Analysis window's Moves tab, for move-by-\n"
                          "move review and analysis with any engine.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(0, 10))
        Button(panel, text="Load Game...", command=self._open_load_game_dialog).pack(fill=X)

    def _open_model_picker(self):
        models = list_models()
        if not models:
            return

        picker = Toplevel(self.root)
        picker.title("Select Model")
        picker.geometry("300x400")
        picker.transient(self.root)
        picker.grab_set()

        Label(picker, text="Select a model:", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        list_frame = Frame(picker)
        list_frame.pack(fill=BOTH, expand=True, padx=10)

        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.configure(command=listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        listbox.pack(side=LEFT, fill=BOTH, expand=True)

        for m in models:
            listbox.insert(END, m)

        current = self.setup_model_var.get()
        if current in models:
            idx = models.index(current)
            listbox.selection_set(idx)
            listbox.see(idx)

        def choose(event=None):
            sel = listbox.curselection()
            if sel:
                self.setup_model_var.set(models[sel[0]])
            picker.destroy()

        listbox.bind("<Double-Button-1>", choose)

        btn_row = Frame(picker)
        btn_row.pack(fill=X, padx=10, pady=10)
        Button(btn_row, text="Select", command=choose).pack(side=RIGHT)
        Button(btn_row, text="Cancel", command=picker.destroy).pack(side=RIGHT, padx=(0, 5))

    def _build_analysis_result_panel(self):
        """The merged Analysis window: a "Position" column (engine-plays/overlay/debug-mode
        controls), a "Search" column (the candidate-move breakdown and, below it, the debug-mode
        playout log -- see _build_search_column; it gets its own column, rather than sharing
        Position's, because both boxes are wide multi-line lists that need real room), and a
        "Moves" column (_build_moves_tab -- the former standalone "Load Game" window's
        move-list/navigation, now living alongside the rest instead of as a separate panel/
        dialog -- see _load_game_for_reference and _jump_to_ply), all three side by side so
        nothing needs clicking through to see another. "End Session / Back to Setup" sits above
        all three, spanning the full width.
        """
        panel = Frame(self.sideFrame)
        self.analysisResultPanel = panel

        Label(panel, text="Session", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 10))
        Button(panel, text="End Session / Back to Setup", command=self.end_analysis_mode).pack(fill=X, pady=(0, 10))

        columns_row = Frame(panel)
        columns_row.pack(fill=BOTH, expand=True)

        position_col = Frame(columns_row)
        position_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        Frame(columns_row, width=1, bg="#999999").pack(side=LEFT, fill=Y)
        search_col = Frame(columns_row)
        search_col.pack(side=LEFT, fill=BOTH, expand=True, padx=8)
        Frame(columns_row, width=1, bg="#999999").pack(side=LEFT, fill=Y)
        moves_col = Frame(columns_row)
        moves_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(8, 0))
        self.movesColumn = moves_col

        Label(position_col, text="Position", font=("Helvetica", 12, "bold"), anchor="w").pack(fill=X)
        Label(search_col, text="Search", font=("Helvetica", 12, "bold"), anchor="w").pack(fill=X)
        Label(moves_col, text="Moves", font=("Helvetica", 12, "bold"), anchor="w").pack(fill=X)

        self._build_position_tab(position_col)
        self._build_search_column(search_col)
        self._build_moves_tab(moves_col)

    def _build_position_tab(self, panel):
        Label(panel, text="Engine plays:", anchor="w").pack(fill=X, pady=(10, 0))
        engine_plays_row = Frame(panel)
        engine_plays_row.pack(fill=X, pady=(2, 10))
        self.engine_plays_black_var = BooleanVar(value=False)
        Checkbutton(engine_plays_row, text="Black", variable=self.engine_plays_black_var,
                    command=self._on_engine_plays_change).pack(side=LEFT)
        self.engine_plays_white_var = BooleanVar(value=False)
        Checkbutton(engine_plays_row, text="White", variable=self.engine_plays_white_var,
                    command=self._on_engine_plays_change).pack(side=LEFT, padx=(10, 0))

        limit_row = Frame(panel)
        limit_row.pack(fill=X, pady=(0, 10))
        Label(limit_row, text="Playout limit:").pack(side=LEFT)
        self.analysis_limit_var = StringVar(value="4000")
        Entry(limit_row, textvariable=self.analysis_limit_var, width=8).pack(side=LEFT, padx=(5, 5))
        Button(limit_row, text="Apply", command=self.apply_analysis_limit).pack(side=LEFT)
        self.analysisPauseButton = Button(limit_row, text="Pause", command=self.toggle_analysis_pause)
        self.analysisPauseButton.pack(side=LEFT, padx=(5, 0))

        Label(panel, text="Board overlay:", anchor="w").pack(fill=X)
        overlay_row = Frame(panel)
        overlay_row.pack(fill=X, pady=(2, 10))
        self.overlay_var = StringVar(value="none")
        for label, value in (("No effect", "none"), ("Policy", "policy"),
                              ("Visits", "visits"), ("Variation", "variation")):
            Radiobutton(overlay_row, text=label, variable=self.overlay_var, value=value,
                        command=self._on_overlay_mode_change).pack(side=LEFT)

        # Territory dots are drawn on top of whichever mode above is active (or on their own,
        # with "No effect") -- an independent checkbox rather than another radio option.
        nn_map_row = Frame(panel)
        nn_map_row.pack(fill=X, pady=(0, 10))
        self.show_territory_var = BooleanVar(value=False)
        Checkbutton(nn_map_row, text="Territory (scoreMap)", variable=self.show_territory_var,
                    command=self._on_overlay_mode_change).pack(side=LEFT)

        self.analysisWinrateLabel = Label(panel, text="Win probability: -", font=("Helvetica", 12), anchor="w")
        self.analysisWinrateLabel.pack(fill=X, pady=5, anchor="w")
        self.analysisInitQLabel = Label(panel, text="Initial value (initQ): -", font=("Helvetica", 12), anchor="w")
        self.analysisInitQLabel.pack(fill=X, pady=5, anchor="w")
        self.analysisScoreExpLabel = Label(panel, text="Score head (initial): -", font=("Helvetica", 12), anchor="w")
        self.analysisScoreExpLabel.pack(fill=X, pady=5, anchor="w")
        self.analysisScoreSearchLabel = Label(panel, text="Score head (search): -", font=("Helvetica", 12), anchor="w")
        self.analysisScoreSearchLabel.pack(fill=X, pady=5, anchor="w")
        self.analysisValueMSELabel = Label(panel, text="Value MSE loss: -", font=("Helvetica", 12), anchor="w")
        self.analysisValueMSELabel.pack(fill=X, pady=5, anchor="w")
        self.analysisPolicyCELabel = Label(panel, text="Policy CE loss: -", font=("Helvetica", 12), anchor="w")
        self.analysisPolicyCELabel.pack(fill=X, pady=5, anchor="w")
        self.analysisVisitsLabel = Label(panel, text="Visits: -", font=("Helvetica", 12), anchor="w")
        self.analysisVisitsLabel.pack(fill=X, pady=5, anchor="w")

        self.debug_mode_var = BooleanVar(value=False)
        Checkbutton(panel, text="Debug mode (log every playout's search path + NN eval)",
                    variable=self.debug_mode_var, command=self._on_debug_mode_change,
                    wraplength=260, justify=LEFT, anchor="w").pack(fill=X, pady=(0, 10))

    def _build_search_column(self, panel):
        """The Search column: candidateFrame (the ordinary Policy/Visits/Variation per-move
        breakdown) on top, and debugFrame (the debug-mode per-playout log) below it -- both
        always visible, stacked in their own column rather than sharing Position's (they're
        both full-width multi-line lists that need real room, which is what made the previous
        single shared box cramped). debugFrame just stays empty while debug mode is off, instead
        of being unpacked/hidden -- so toggling debug mode never resizes or reflows this column,
        only fills in (or clears) its lower box.
        """
        self.candidateFrame = Frame(panel)
        self.candidateListHeader = Label(self.candidateFrame, text="Candidate moves, by visit count:",
                                          font=("Helvetica", 10, "bold"), anchor="w",
                                          wraplength=260, justify=LEFT)
        self.candidateListHeader.pack(fill=X)

        # Two interchangeable header widgets stacked in the same slot (see
        # _update_candidate_listbox, which packs whichever matches the current overlay mode):
        # candidateColumnHeaderFrame's per-column labels are clickable to sort by that field
        # (visits/policy/winrate/Q/pref) in Policy/Visits overlay mode; candidateVariationHeader
        # is the plain single-line label Variation mode used already, unsortable since that mode
        # lists move sequences rather than a single numeric stat.
        self.candidateColumnHeaderFrame = Frame(self.candidateFrame)
        self._candidate_sort_labels = {}
        self._candidate_header_text = {}
        for key, text, width in _CANDIDATE_COLUMNS:
            self._candidate_header_text[key] = text.ljust(width)
            lbl = Label(self.candidateColumnHeaderFrame, text=text.ljust(width),
                        font=("Courier", 9), fg="#555555", anchor="w")
            lbl.pack(side=LEFT)
            if key is not None:
                lbl.configure(cursor="hand2")
                lbl.bind("<Button-1>", lambda e, k=key: self._on_candidate_sort_header_click(k))
            self._candidate_sort_labels[key] = lbl
        self.candidateColumnHeaderFrame.pack(fill=X, pady=(2, 2))

        self.candidateVariationHeader = Label(self.candidateFrame, text="expected continuation",
                                               font=("Courier", 9), fg="#555555", anchor="w")

        candidate_list_frame = Frame(self.candidateFrame)
        candidate_list_frame.pack(fill=BOTH, expand=True)
        candidate_scrollbar = Scrollbar(candidate_list_frame, orient=VERTICAL)
        self.analysisListbox = Listbox(candidate_list_frame, yscrollcommand=candidate_scrollbar.set,
                                        font=("Courier", 10), height=20, exportselection=False)
        candidate_scrollbar.configure(command=self.analysisListbox.yview)
        candidate_scrollbar.pack(side=RIGHT, fill=Y)
        self.analysisListbox.pack(side=LEFT, fill=BOTH, expand=True)
        self.analysisListbox.bind("<<ListboxSelect>>", self._on_candidate_listbox_select)
        self.candidateFrame.pack(side=TOP, fill=BOTH, expand=True)

        self.debugFrame = Frame(panel)
        self.debugFrame.pack(side=TOP, fill=BOTH, expand=True, pady=(10, 0))
        Frame(self.debugFrame, height=1, bg="#999999").pack(fill=X, pady=(0, 8))
        Label(self.debugFrame, text="Playouts, in search order (debug mode) -- select one to\n"
                                     "trace its path on the board:",
              font=("Helvetica", 10, "bold"), anchor="w", wraplength=260, justify=LEFT).pack(fill=X)

        debug_header_row = Frame(self.debugFrame)
        debug_header_row.pack(fill=X, pady=(2, 2))
        Label(debug_header_row, text="path (row+col per move, e.g. \"24 28\")",
              font=("Courier", 9), fg="#555555", anchor="w").pack(side=LEFT, fill=X, expand=True)
        Label(debug_header_row, text="#     winp   score", font=("Courier", 9), fg="#555555",
              anchor="w").pack(side=RIGHT)

        # The path column gets its own small listbox with its own horizontal scrollbar, kept in
        # step (vertically) with a second, fixed-width listbox holding "# winp score" -- so a
        # long search path scrolls within its own little window instead of shoving winp/score
        # off to the right where they'd need a wide window to stay visible (that was the bug:
        # a single Listbox with one long fixed-width line per row, no horizontal scrollbar).
        debug_list_frame = Frame(self.debugFrame)
        debug_list_frame.pack(fill=BOTH, expand=True)

        debug_vscrollbar = Scrollbar(debug_list_frame, orient=VERTICAL)
        debug_vscrollbar.pack(side=RIGHT, fill=Y)

        stat_frame = Frame(debug_list_frame)
        stat_frame.pack(side=RIGHT, fill=Y)
        self.debugListbox = Listbox(stat_frame, font=("Courier", 9), height=20, width=18,
                                     exportselection=False)
        self.debugListbox.pack(fill=Y, expand=True)

        path_frame = Frame(debug_list_frame)
        path_frame.pack(side=LEFT, fill=BOTH, expand=True)
        debug_hscrollbar = Scrollbar(path_frame, orient=HORIZONTAL)
        debug_hscrollbar.pack(side=BOTTOM, fill=X)
        self.debugPathListbox = Listbox(path_frame, xscrollcommand=debug_hscrollbar.set,
                                         font=("Courier", 9), height=20, exportselection=False)
        debug_hscrollbar.configure(command=self.debugPathListbox.xview)
        self.debugPathListbox.pack(side=LEFT, fill=BOTH, expand=True)

        # Keep the two listboxes' vertical scroll positions and selections in lock-step, as if
        # they were columns of one widget. _debug_sync_guard prevents the yview_moveto() call
        # below from re-triggering this same handler for the listbox it's synchronizing.
        self._debug_sync_guard = False

        def sync_yview(source, *args):
            debug_vscrollbar.set(*args)
            if self._debug_sync_guard:
                return
            self._debug_sync_guard = True
            try:
                other = self.debugListbox if source is self.debugPathListbox else self.debugPathListbox
                other.yview_moveto(args[0])
            finally:
                self._debug_sync_guard = False

        self.debugPathListbox.configure(yscrollcommand=lambda *a: sync_yview(self.debugPathListbox, *a))
        self.debugListbox.configure(yscrollcommand=lambda *a: sync_yview(self.debugListbox, *a))
        debug_vscrollbar.configure(command=self._debug_listboxes_yview)

        for lb in (self.debugPathListbox, self.debugListbox):
            lb.bind("<<ListboxSelect>>", self._on_debug_listbox_select)
            lb.bind("<MouseWheel>", self._on_debug_list_mousewheel)
            lb.bind("<Button-4>", self._on_debug_list_mousewheel)
            lb.bind("<Button-5>", self._on_debug_list_mousewheel)

    def _build_moves_tab(self, panel):
        """The former standalone "Load Game for Review" window's content, now a tab of the
        merged Analysis window: an optional loaded game's info, position navigation, the move
        table itself (see _update_moves_tab for the B eval/W eval/Current eval convention), and
        a way to attach a live engine at whatever position is currently shown -- reusing
        start_session exactly as the setup panel's own "Start Session" does.
        """
        self.movesInfoLabel = Label(panel, text="", font=("Helvetica", 11), anchor="w",
                                     wraplength=260, justify=LEFT)
        self.movesInfoLabel.pack(fill=X, pady=(10, 8))

        Button(panel, text="Load Game...", command=self._open_load_game_dialog).pack(fill=X, pady=(0, 8))

        nav_row = Frame(panel)
        nav_row.pack(fill=X, pady=(0, 2))
        Button(nav_row, text="|<", width=3, command=lambda: self._jump_to_ply(0)).pack(side=LEFT)
        Button(nav_row, text="<", width=3,
               command=lambda: self._jump_to_ply(self.review_index - 1)).pack(side=LEFT)
        Button(nav_row, text=">", width=3,
               command=lambda: self._jump_to_ply(self.review_index + 1)).pack(side=LEFT)
        Button(nav_row, text=">|", width=3,
               command=lambda: self._jump_to_ply(len(self._current_moves_coords()))).pack(side=LEFT)
        self.movesPositionLabel = Label(nav_row, text="", anchor="w")
        self.movesPositionLabel.pack(side=LEFT, padx=(10, 0))

        Label(panel, text="Moves (click a row to jump there):", font=("Helvetica", 10, "bold"),
              anchor="w").pack(fill=X, pady=(8, 0))
        tree_frame = Frame(panel)
        tree_frame.pack(fill=BOTH, expand=True, pady=(2, 10))
        tree_scrollbar = Scrollbar(tree_frame, orient=VERTICAL)
        columns = ("ply", "move", "b_eval", "w_eval", "cur_eval")
        self.movesTree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                       yscrollcommand=tree_scrollbar.set, selectmode="browse")
        for col, heading, width in (("ply", "#", 40), ("move", "Move", 75), ("b_eval", "B eval", 70),
                                     ("w_eval", "W eval", 70), ("cur_eval", "Current", 75)):
            self.movesTree.heading(col, text=heading)
            self.movesTree.column(col, width=width, anchor=CENTER, stretch=(col == "move"))
        tree_scrollbar.configure(command=self.movesTree.yview)
        tree_scrollbar.pack(side=RIGHT, fill=Y)
        self.movesTree.pack(side=LEFT, fill=BOTH, expand=True)
        self.movesTree.bind("<<TreeviewSelect>>", self._on_moves_tree_select)

        Frame(panel, height=1, bg="#999999").pack(fill=X, pady=10)

        Label(panel, text="Start analyzing this position:", font=("Helvetica", 11, "bold"),
              anchor="w").pack(fill=X)
        Label(panel, text="With any engine you like -- not necessarily the one(s) that played\n"
                          "a loaded game, if one is loaded.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(2, 8))

        model_row = Frame(panel)
        model_row.pack(fill=X, pady=(0, 8))
        Label(model_row, textvariable=self.setup_model_var, anchor="w",
              relief=SUNKEN, bg="white", padx=5).pack(side=LEFT, fill=X, expand=True)
        Button(model_row, text="Choose...", command=self._open_model_picker).pack(side=LEFT, padx=(5, 0))

        models = list_models()
        self.moves_analyze_button = Button(panel, text="Start Analyzing", command=self._start_analysis_from_panel)
        self.moves_analyze_button.pack(fill=X)
        if not models:
            self.moves_analyze_button.configure(state=DISABLED)

    def _show_panel(self, panel):
        for p in (self.setupPanel, self.analysisResultPanel):
            p.pack_forget()
        panel.pack(fill=BOTH, expand=True)

    def start(self):
        """Open the main window with a freely playable board and the setup panel."""
        self.boardFrame.grid(row=0, column=0, sticky="n")
        self.sideFrame.grid(row=0, column=1, sticky="n", padx=20, pady=20)
        self.canvas.pack()
        self.passButton.pack()
        self.passButton.configure(command=self.on_pass)
        self.statusLabel.pack()

        for i in range(self.boardSize):
            self.canvas.create_line(i * self.delta + self.m, self.m, i * self.delta + self.m,
                               (self.boardSize - 1) * self.delta + self.m, tags="grid")
            self.canvas.create_line(self.m, i * self.delta + self.m, (self.boardSize - 1) * self.delta + self.m,
                               i * self.delta + self.m, tags="grid")

        self._draw_star_points()

        for i in self.neutral:
            self.canvas.create_image(self.m + i[0] * self.delta - self.stone_size / 2,
                                     self.m + i[1] * self.delta - self.stone_size / 2,
                        anchor=NW, image=self.stones[2], tags="neutral")
            self.on_stone[i[0]][i[1]] = True

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                self.cord[i][j] = (self.m + i * self.delta, self.m + j * self.delta)

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._show_panel(self.setupPanel)
        self._update_status()

    def _draw_star_points(self):
        if self.boardSize < 7:
            return
        offsets = sorted({2, self.boardSize - 3})
        points = {(x, y) for x in offsets for y in offsets}
        if self.boardSize % 2 == 1:
            points.add((self.boardSize // 2, self.boardSize // 2))

        r = max(3, int(self.delta * 0.08))
        for x, y in points:
            cx = self.m + x * self.delta
            cy = self.m + y * self.delta
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="black", outline="black", tags="star")

    def _place_stone(self, x, y, color_idx):
        x_loc = self.m + x * self.delta
        y_loc = self.m + y * self.delta
        self.canvas.create_image(x_loc - self.stone_size / 2, y_loc - self.stone_size / 2,
                                 anchor=NW, image=self.stones[color_idx], tags="stone")
        self.on_stone[x][y] = True

    def _update_status(self):
        if self.game_over:
            return
        if not self.analysis_mode:
            self.statusLabel.configure(text="Black to move" if self.turn == 0 else "White to move")
            return
        color = "Black" if self.turn == 0 else "White"
        if self._is_engine_turn():
            self.statusLabel.configure(text=f"Engine is thinking... ({color} to move)")
        else:
            self.statusLabel.configure(text=f"Your turn ({color} to move)")

    def _is_engine_turn(self):
        """Whether the engine has been told to auto-play whoever is currently to move --
        independent of analysis_mode's on/off state, this is itself freely toggled at any
        time via the "Engine plays" checkboxes, so a session can move between the engine
        playing a side, the human playing it, and pure hands-on analysis of both sides.
        Always false while a loaded game is being browsed (review_mode) -- auto-play only
        makes sense for a live game actually being played out, never for stepping through a
        fixed, already-finished recorded one, regardless of what the checkboxes (left over
        from some earlier, unrelated session) happen to show."""
        if not self.analysis_mode or self.review_mode:
            return False
        if self.turn == 0:
            return self.engine_plays_black_var.get()
        return self.engine_plays_white_var.get()

    def _commit_move(self, x, y):
        """Applies one move -- a stone at (x, y), or a pass when (x, y) == (boardSize, 0) --
        to the local rule engine and, if a session is live, forwards it to the analysis
        engine and re-requests analysis for the resulting position. Shared by human clicks/
        pass and the engine's own auto-played moves (_auto_play_move); callers are
        responsible for any legality/turn gating before calling this.

        If search was paused, moving doesn't resume it: the move still jumps the analysis
        engine's tree to that child (send_play), but the analysis re-request asks for only
        _PAUSED_MOVE_MIN_VISITS playouts -- per the "analyze" protocol's cumulative-target
        semantics, a no-op for any child that already has more visits than that, so a child
        with real prior search just reports its current stats as-is; only a barely-touched
        child gets the couple of playouts it needs to show a real (non-blank) per-move
        breakdown instead of one big resumed search. Pause stays engaged either way.

        Otherwise (not paused), the Playout limit field is meant as "how many playouts to spend
        on this move", not "search until root visits reach this number" -- but "analyze N" is a
        cumulative target on the *engine's* side (see modelcompare.cpp), and the tree persists
        across moves, so the position just jumped to typically already carries over some visits
        from having been explored as a child during the previous move's search. Left alone,
        that carry-over would silently eat into the requested budget (or, on a well-explored
        child, make the request an outright no-op). So the target sent is the carried-over visit
        count (matched["visits"], from the same per-move breakdown used for session_move_log
        above -- exactly what the child's visits will be immediately after send_play) plus the
        configured limit, to land on "limit *more* playouts" instead. A proven win/loss still
        cuts search short regardless of the target (rootForcedState() in the engine's own search
        loop) -- that exception needs no special-casing here.
        """
        is_pass = (x == self.boardSize and y == 0)

        # Logged regardless of session state (see session_move_log's own comment in __init__) --
        # this is the single mutation point for both human clicks/pass and engine auto-play
        # (_auto_play_move), so it's the natural place to record "the entire sequence of the
        # game" that the Moves tab later browses (see _current_moves_list). Captured before the
        # move is applied below, since last_analysis_result describes the position we're still
        # in (the one the move being committed right now was chosen from/for).
        matched = None
        if self.analysis_mode and self.last_analysis_result is not None:
            matched = next((mv for mv in self.last_analysis_result.get("moves", [])
                             if mv["move"] == (x, y)), None)
        root = self.last_analysis_result if self.analysis_mode else None
        ply = len(self.rule.seq)

        # Locks in the Moves tab's "Current evaluation" for the position being left, synchronously
        # and unconditionally -- rather than relying solely on a streamed analysis result arriving
        # for it before poll_analysis's next tick, which can race with the "send_play" a few lines
        # down: that resets the engine onto the *next* position, and any result already in flight
        # for this one that lands in the queue afterward would otherwise get mis-attributed to the
        # new ply (see _update_analysis_result_panel, which keys purely off len(self.rule.seq) at
        # whenever a result happens to be drained). last_analysis_result is already trusted for
        # this same position just above (matched/root), so reusing it here is exactly "whatever
        # was last known about the position now being left."
        if root is not None:
            self.position_eval_log[ply] = {
                "winrate": root.get("winrate"),
                "visits": root.get("visits"),
                "forced": root.get("forced"),
            }

        if ply < len(self.session_move_log):
            # Only reachable via _jump_to_ply rewinding this session's own history (review_mode
            # blocks clicks entirely, so a loaded game's fixed moves never hit this path) and
            # then playing a move that diverges from what was there before -- drop the old
            # "future" so session_move_log stays exactly in step with self.rule.seq, index for
            # index, the way _current_moves_list/_jump_to_ply both assume.
            self.session_move_log = self.session_move_log[:ply]
        self.session_move_log.append({
            "ply": ply,
            "color": "black" if self.turn == 0 else "white",
            "mover": "engine" if self._is_engine_turn() else "human",
            "move": None if is_pass else [x, y],
            "visits": matched["visits"] if matched else None,
            "prior": matched["prior"] if matched else None,
            "move_winrate": matched["winrate"] if matched else None,
            "move_q": matched["q"] if matched else None,
            "root_winrate": (root or {}).get("winrate"),
            "root_visits": (root or {}).get("visits"),
            "root_initQ": (root or {}).get("initQ"),
            "root_scoreExp": (root or {}).get("scoreExp"),
            "root_scoreSearch": (root or {}).get("scoreSearch"),
        })
        # review_record is never set here in practice -- review_mode (loaded game) blocks both
        # clicks/pass and auto-play (see on_click/on_pass/_is_engine_turn) -- but update
        # unconditionally regardless, since it's cheap and this is the single mutation point.
        self.review_index = ply + 1
        self._update_moves_tab()

        if is_pass:
            result = self.rule.make_move(-1, 0)
        else:
            self._place_stone(x, y, self.turn)
            self.turn = 1 - self.turn
            result = self.rule.make_move(x, y)

        if self.analysis_mode:
            self.analysisEngine.send_play(x, y)
            # New position -> new turn: reset the Visits-label baseline to this position's
            # carried-over visit count (see analysis_turn_base_visits' comment in __init__).
            base_visits = int(matched["visits"]) if matched else 0
            self.analysis_turn_base_visits = base_visits
            if self.analysis_paused:
                target = _PAUSED_MOVE_MIN_VISITS
                self.analysis_turn_requested = max(0, target - base_visits)
                self.analysisEngine.request_analysis(target)
            else:
                self._request_more_playouts()
            self.selected_variation_move = None
            # A new position means the old playout log's tree no longer applies (it described
            # search on the position we just left); AnalysisEngine doesn't track this on its
            # own end at all now, so clearing our own accumulated copy is entirely on us.
            self.playout_log = []
            self.selected_playout_index = None
            if self.debug_mode_var.get():
                self._update_debug_listbox()
            if not self.analysis_paused:
                self.analysisPauseButton.configure(text="Pause")

        if is_pass:
            if result == -2:
                self.on_end(-2)
                return
            self.turn = 1 - self.turn
            self._update_status()
        else:
            if result == 1 or result == -1 or result == -2:
                self.on_end(result * self.rule.turn)
            else:
                self._update_status()

    def on_click(self, event):
        if self.game_over or self.review_mode:
            return
        if self._is_engine_turn():
            return  # the engine auto-plays this color right now; ignore board clicks

        x_cord = int((event.x - self.m) / self.delta + 0.5)
        y_cord = int((event.y - self.m) / self.delta + 0.5)
        x_loc = self.m + x_cord * self.delta
        y_loc = self.m + y_cord * self.delta

        if x_cord < 0 or x_cord >= self.boardSize or y_cord < 0 or y_cord >= self.boardSize:
            return

        if not (abs(event.x - x_loc) < self.cl and abs(event.y - y_loc) < self.cl):
            return
        if self.on_stone[x_cord][y_cord]:
            return
        # playing into the opponent's already-settled territory is illegal
        if self.rule.terr_board[x_cord + 1][y_cord + 1] * self.rule.turn == -1:
            return

        self._commit_move(x_cord, y_cord)

    def on_pass(self):
        if self.game_over or self.review_mode:
            return
        if self._is_engine_turn():
            return  # the engine auto-plays this color right now; ignore the pass button

        self._commit_move(self.boardSize, 0)

    def _reset_debug_mode(self):
        """Local GUI-side reset to match a fresh (or about-to-be-replaced) AnalysisEngine
        process, which always starts with debug mode off -- used whenever a session ends or a
        new one starts, so a checkbox left ticked from a previous session doesn't look enabled
        while actually talking to a process that was never told to turn debug mode on.
        """
        self.debug_mode_var.set(False)
        self.playout_log = []
        self.selected_playout_index = None
        self.debugPathListbox.delete(0, END)
        self.debugListbox.delete(0, END)

    def reset_board(self):
        """Clear the board, end any live session, and start a fresh local (no engine) game."""
        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None
        self.canvas.delete("stone")
        self.canvas.delete("move_overlay")
        self.last_analysis_result = None
        self.analysis_turn_base_visits = 0
        self.analysis_turn_requested = 0
        self.selected_variation_move = None
        self._reset_debug_mode()
        self.analysis_paused = False
        self._auto_move_seq_len = -1
        self.analysisPauseButton.configure(text="Pause")
        self.rule = RuleManager()
        self.on_stone = [[False for _ in range(self.boardSize)] for _ in range(self.boardSize)]
        self.turn = 0
        self.game_over = False
        self.analysis_mode = False
        self.engine_plays_black_var.set(False)
        self.engine_plays_white_var.set(False)
        self.session_move_log = []
        self.session_used_engine = False
        self.session_model = None
        self.review_mode = False
        self.review_record = None
        self.review_moves = []
        self.review_index = 0
        self.position_eval_log = {}
        self.game_over_ply = None
        self.game_over_text = None
        self.statusLabel.configure(text="")
        self._show_panel(self.setupPanel)
        self._update_status()
        self._update_moves_tab()

    def start_session(self, model=None):
        """Launch (or restart) the analysis engine on the position currently on the board,
        and set which side(s) -- if any -- it auto-plays for, per the setup panel's choice.
        Since this doesn't clear the board first, it doubles as both "analyze this position"
        and "play a game against the engine from here"; which one it feels like is entirely
        down to the Engine plays checkboxes, freely changeable afterwards from the session
        panel without ending the session. Also reachable from the Moves tab
        (_start_analysis_from_panel), where self.rule already holds whatever position was being
        browsed (possibly review_mode-locked to a loaded game) -- game_over is reset
        unconditionally since browsing can leave it set for a finished game's last position, and
        position_eval_log is cleared since a freshly (re)started engine's opinions of positions
        shouldn't mix with whatever an earlier engine/session already recorded there.
        """
        if model is None:
            model = self.setup_model_var.get()
        if not model:
            return

        if self.analysisEngine is not None:
            self.analysisEngine.close()

        self.analysisEngine = AnalysisEngine(model, build_dir=BUILD_DIR)
        self.analysis_mode = True
        self.session_used_engine = True
        self.session_model = model
        self.game_over = False
        self.position_eval_log = {}

        if not self.review_mode:
            # A loaded/locked game never auto-plays (see _is_engine_turn) regardless of these --
            # leave them (and whatever a prior, unrelated session set them to) alone rather than
            # re-deriving from the setup panel's choice, which wasn't made with this game in mind.
            choice = self.setup_color_var.get()
            self.engine_plays_black_var.set(choice == "engine_black")
            self.engine_plays_white_var.set(choice == "engine_white")

        self._sync_analysis_position()

        self._show_panel(self.analysisResultPanel)
        self._update_status()
        self._update_moves_tab()
        self.root.after(100, self.poll_analysis)

    def _sync_analysis_position(self):
        """Replay the current game's move sequence into the analysis engine, then leave it
        paused -- analysis only actually starts once the user presses Resume."""
        self.selected_variation_move = None
        self._reset_debug_mode()
        self._auto_move_seq_len = -1
        self.analysis_paused = True
        self.analysisPauseButton.configure(text="Resume")
        # A plain replay (send_play with no "analyze" in between) leaves every node on the
        # replayed path unexplored -- no carryover, unlike a move committed during live search
        # (see _commit_move) -- and last_analysis_result may still be describing whatever
        # position/session preceded this sync, so it can't be trusted as this turn's starting
        # point either. New turn, clean baseline.
        self.last_analysis_result = None
        self.analysis_turn_base_visits = 0
        self.analysis_turn_requested = 0
        self.analysisEngine.send_reset()
        for x, y in self.rule.seq:
            if x == -1 or y == -1:
                self.analysisEngine.send_play(self.boardSize, 0)
            else:
                self.analysisEngine.send_play(x, y)

    def _get_analysis_limit(self):
        try:
            return max(1, int(self.analysis_limit_var.get()))
        except ValueError:
            return 4000

    def _request_more_playouts(self, extra=None):
        """Sends "analyze" for a target of analysis_turn_base_visits + `extra` (default: the
        Playout limit field) -- "analyze N"'s target is an absolute cumulative visit count on
        the engine's side (see modelcompare.cpp), so this recomputes it fresh from the field
        every time, rather than piling `extra` on top of however far search has already gotten.
        That distinction matters once this is called more than once for the same position
        (successive Apply/Resume presses): recomputing fresh means nudging the field from 4000
        to 4001 and hitting Apply runs exactly one more playout (the target rises by exactly 1,
        offset by the same fixed base both times) -- stacking `extra` on top of current visits
        each press would instead run another whole 4001. Leaving the field untouched and
        pressing Apply again is correctly a no-op (same target as last time, already reached).
        analysis_turn_requested tracks the field value this last sent, so the Visits label and
        _compute_analysis_status compare against the same target this actually requested (see
        analysis_turn_base_visits' comment in __init__).
        """
        if extra is None:
            extra = self._get_analysis_limit()
        self.analysis_turn_requested = extra
        self.analysisEngine.request_analysis(self.analysis_turn_base_visits + extra)

    def apply_analysis_limit(self):
        """Re-requests analysis at this turn's carried-over baseline plus the Playout limit
        field's current value (_request_more_playouts) -- so raising the limit after search
        already finished and hitting Apply resumes search by exactly the difference (bump it
        from 4000 to 4001 and one more playout runs, not another 4001), rather than either
        no-op'ing (the old flat-target bug) or overshooting by the field's full value each press.
        """
        if not self.analysis_mode or self.analysisEngine is None:
            return
        self.analysis_paused = False
        self.analysisPauseButton.configure(text="Pause")
        self._request_more_playouts()

    def toggle_analysis_pause(self):
        """Pause halts the engine's in-progress search after its current chunk, keeping
        the tree as-is; Resume re-requests analysis at this turn's baseline plus the
        configured limit (_request_more_playouts), same as Apply.
        """
        if not self.analysis_mode or self.analysisEngine is None:
            return
        if self.analysis_paused:
            self.analysis_paused = False
            self.analysisPauseButton.configure(text="Pause")
            self._request_more_playouts()
        else:
            self.analysis_paused = True
            self.analysisPauseButton.configure(text="Resume")
            self.analysisEngine.send_pause()

    def end_analysis_mode(self):
        self.analysis_mode = False
        self.engine_plays_black_var.set(False)
        self.engine_plays_white_var.set(False)
        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None
        self.canvas.delete("move_overlay")
        self.last_analysis_result = None
        self._reset_debug_mode()
        self._show_panel(self.setupPanel)

    def poll_analysis(self):
        # game_over is ignored while browsing a loaded game (review_mode): _jump_to_ply can
        # leave it set (or stale) for that game's final position, but analysis on a loaded game
        # should keep running regardless of where in its history is currently being viewed.
        if not self.analysis_mode or (self.game_over and not self.review_mode):
            return
        latest = None
        while True:
            r = self.analysisEngine.get_result_nowait()
            if r is None:
                break
            latest = r

        # Debug-mode playout records stream independently of the above (see AnalysisEngine.
        # playouts) -- the engine stores none of this itself, so accumulating the growing list
        # a debug session looks at is entirely on us; drain whatever's arrived since last tick.
        got_playout = False
        while True:
            p = self.analysisEngine.get_playout_nowait()
            if p is None:
                break
            self.playout_log.append(p)
            got_playout = True

        if latest is not None:
            self._update_analysis_result_panel(latest)
        if got_playout and self.debug_mode_var.get():
            self._update_debug_listbox()
        if self.analysis_mode and not self.game_over:
            self.root.after(200, self.poll_analysis)

    def _on_overlay_mode_change(self):
        if self.last_analysis_result is not None:
            self._draw_move_overlay(self.last_analysis_result)
            self._update_candidate_listbox(self.last_analysis_result)
        else:
            self.canvas.delete("move_overlay")

    def _draw_move_overlay(self, result):
        self.canvas.delete("move_overlay")

        # Territory dots are an independent checkbox, layered under whichever candidate-move
        # mode below is also active -- drawn unconditionally so it still shows up even with
        # overlay mode "none".
        self._draw_territory_overlay(result)

        mode = self.overlay_var.get()
        moves = result.get("moves", [])
        if mode == "variation":
            # A selected playout (debug mode) takes over the Variation overlay -- its actual
            # search path, instead of the top/selected candidate's engine-line continuation.
            if self.debug_mode_var.get() and self.selected_playout_index is not None:
                self._draw_playout_path_overlay()
            elif moves:
                self._draw_variation_overlay(moves, result.get("forced") or 0)
        elif mode in ("policy", "visits") and moves:
            if mode == "policy":
                # prior is already a probability (fraction of 1); use it directly as the "share".
                get_val = lambda mv: mv["prior"]
                get_share = lambda mv: mv["prior"]
                total_visits = None
            else:
                total_visits = result.get("visits") or 0
                get_val = lambda mv: mv["visits"]
                get_share = lambda mv: mv["visits"] / total_visits if total_visits else 0.0

            max_val = max((get_val(mv) for mv in moves), default=0.0)
            forced = result.get("forced") or 0
            if max_val > 0 and (mode != "visits" or total_visits > 0):
                r = int(self.stone_size * 0.42)
                font_size = max(7, int(self.stone_size * 0.16))
                for mv in moves:
                    if get_share(mv) < 0.01:  # hide anything under 1% probability/visit share -- mostly noise
                        continue
                    val = get_val(mv)
                    x, y = mv["move"]
                    if x < 0 or x >= self.boardSize or y < 0 or y >= self.boardSize or self.on_stone[x][y]:
                        continue

                    # Once forced, "moves" holds exactly this one candidate -- the proven
                    # winning (or best-delaying) move -- so on the Visits overlay it gets a flat
                    # categorical color instead of the usual visit-share gradient, which would
                    # otherwise render it identically to any other move that happened to hold
                    # 100% of the visits without actually being proven.
                    if mode == "visits" and forced > 0:
                        color = FORCED_WIN_COLOR
                    elif mode == "visits" and forced < 0:
                        color = FORCED_LOSS_COLOR
                    else:
                        color = _blend_hex(BOARD_BG, OVERLAY_COLOR, val / max_val)
                    text_color = "#ffffff" if _luminance(color) < 140 else "#000000"
                    star = " ★" if (mode == "visits" and forced > 0) else ""
                    label = f"{val * 100:.0f}%" if mode == "policy" else f"{int(val)}{star}"

                    cx = self.m + x * self.delta
                    cy = self.m + y * self.delta
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="",
                                             tags=("move_overlay", "move_overlay_shape"))
                    self.canvas.create_text(cx, cy, text=label, fill=text_color,
                                             font=("Helvetica", font_size, "bold"),
                                             tags=("move_overlay", "move_overlay_text"))

                    q_val = mv.get("q")
                    if q_val is not None:
                        self.canvas.create_text(cx, cy + r + max(8, int(font_size * 0.7)),
                                                 text=f"{q_val:+.2f}", fill="#1a1a1a",
                                                 font=("Helvetica", max(6, font_size - 2)),
                                                 tags=("move_overlay", "move_overlay_text"))

        # stacking order (bottom to top): territory dots / overlay circles, grid/star (board
        # coordinates), stones, overlay numbers. Sink the shapes below everything already on the
        # board (grid/star/stones) instead of raising grid/star, which would otherwise climb
        # above stones too; keep text on top of all of it so labels stay legible even over a
        # star point.
        self.canvas.tag_lower("move_overlay_shape")
        self.canvas.tag_raise("move_overlay_text")

    def _draw_territory_overlay(self, result):
        """KataGo-style ownership dots: a small black dot on points the network expects to end
        up as Black territory, white (outlined) dot for White -- dot radius scales with the
        network's confidence (|scoreMap| toward the tanh range's ends). Only meaningful at
        currently-empty points; a point with a stone on it isn't "territory".

        scoreMap's sign is relative to whoever is to move (same convention as the win-probability
        head), not an absolute Black/White sign -- calculateQ in PMCTS.cpp combines it directly
        with a "-1 = current turn's own point" convention. So when White is to move, the raw sign
        has to be flipped back to an absolute Black/White reading before picking dot color.
        """
        if not self.show_territory_var.get():
            return
        score_map = result.get("scoreMap")
        if not score_map:
            return

        turn_sign = 1 if self.turn == 0 else -1  # self.turn: 0 = Black, 1 = White to move
        max_r = max(3, int(self.stone_size * 0.20))
        for i, val in enumerate(score_map):
            if abs(val) < 0.08:  # too uncertain to bother marking
                continue
            x, y = divmod(i, self.boardSize)
            if x >= self.boardSize or y >= self.boardSize or self.on_stone[x][y]:
                continue

            abs_val = val * turn_sign  # canonicalize to absolute Black(-)/White(+)
            rad = max(2, int(max_r * min(abs(abs_val), 1.0)))
            fill = "#111111" if abs_val < 0 else "#f5f5f5"  # < 0 -> Black, > 0 -> White
            cx = self.m + x * self.delta
            cy = self.m + y * self.delta
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=fill,
                                     outline="#000000", tags=("move_overlay", "move_overlay_shape"))

    def _draw_variation_overlay(self, moves, forced=0):
        """KataGo-style principal-variation overlay: numbered stone-colored markers tracing the
        expected continuation for the selected candidate (or the top/highest-visit move if none
        has been clicked in the list yet), alternating color by ply.

        Once forced (proven win/loss), "moves" holds exactly the one proven move (see
        Analysis::printAnalysis), so it's already both the highest-visit AND the only candidate --
        "target" below picks it out the same way it always does, with no special-casing needed
        to make it come first. highlight_first then just makes that fact visible on the board:
        marker #1 (the winning move itself) gets FORCED_WIN_COLOR's ring instead of black,
        instead of looking like an ordinary variation.
        """
        target = None
        if self.selected_variation_move is not None:
            target = next((mv for mv in moves if mv["move"] == self.selected_variation_move
                            and mv["visits"] > 0), None)
        if target is None:
            target = max(moves, key=lambda mv: mv["visits"])
        if target["visits"] <= 0:
            return

        seq = [target["move"]] + target.get("variation", [])
        self._draw_numbered_path_overlay(seq, highlight_first=(forced > 0))

    def _draw_playout_path_overlay(self):
        """Debug mode's Variation-overlay override: the selected playout's actual root->leaf
        search path (from our own accumulated playout_log -- see poll_analysis), drawn the same
        way as the ordinary engine-line variation above -- just from a different, per-playout
        source."""
        if self.selected_playout_index is None or self.selected_playout_index >= len(self.playout_log):
            return
        self._draw_numbered_path_overlay(self.playout_log[self.selected_playout_index]["path"])

    def _draw_numbered_path_overlay(self, seq, highlight_first=False):
        """Numbered stone-colored markers tracing a move sequence starting from whoever is
        currently to move, alternating color by ply -- the shared drawing routine behind both
        the ordinary engine-line variation and debug mode's per-playout path. highlight_first
        rings marker #1 in FORCED_WIN_COLOR instead of black -- used only for a proven winning
        move (see _draw_variation_overlay), never for an ordinary/debug-playout path."""
        if not seq:
            return

        r = int(self.stone_size * 0.42)
        font_size = max(7, int(self.stone_size * 0.32))
        mover = self.turn  # 0 = black, 1 = white -- whoever is to move in the analyzed position

        for idx, (x, y) in enumerate(seq):
            if x < 0 or x >= self.boardSize or y < 0 or y >= self.boardSize or self.on_stone[x][y]:
                continue

            black_to_play = (mover + idx) % 2 == 0
            fill = "#111111" if black_to_play else "#f2f2f2"
            text_color = "#ffffff" if black_to_play else "#000000"
            outline = FORCED_WIN_COLOR if (highlight_first and idx == 0) else "#000000"
            outline_width = 3 if (highlight_first and idx == 0) else 1

            cx = self.m + x * self.delta
            cy = self.m + y * self.delta
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline,
                                     width=outline_width, tags=("move_overlay", "move_overlay_shape"))
            self.canvas.create_text(cx, cy, text=str(idx + 1), fill=text_color,
                                     font=("Helvetica", font_size, "bold"),
                                     tags=("move_overlay", "move_overlay_text"))

        self.canvas.tag_lower("move_overlay_shape")
        self.canvas.tag_raise("move_overlay_text")

    def _update_analysis_result_panel(self, result):
        self.last_analysis_result = result
        self._draw_move_overlay(result)

        # The Moves tab's "Current evaluation" column (see _update_moves_tab): every streamed
        # result updates the entry for whatever ply is currently on the board (len(self.rule.
        # seq) -- the number of moves already played, matching session_move_log's own "ply"
        # convention), so re-visiting a former position (_jump_to_ply) during live analysis
        # refreshes its entry here exactly the same way as an ordinary new move does.
        ply = len(self.rule.seq)
        self.position_eval_log[ply] = {
            "winrate": result.get("winrate"),
            "visits": result.get("visits"),
            "forced": result.get("forced"),
        }
        self._update_moves_tab()

        winrate = result.get("winrate")
        if winrate is not None:
            pct = (winrate + 1) / 2 * 100
            self.analysisWinrateLabel.configure(text=f"Win probability: {pct:.1f}%")

        init_q = result.get("initQ")
        if init_q is not None:
            self.analysisInitQLabel.configure(text=f"Initial value (initQ): {init_q:+.4f}")

        # scoreExp/scoreSearch mirror initQ/winrate for the score head: scoreExp is the network's
        # one-shot raw guess the first time this position was evaluated, unmoved by search;
        # scoreSearch is the same figure refined by search (root->S, backed up every playout the
        # same way root->Wp is) -- so it's expected to differ from scoreExp once visits pick up,
        # same as winrate typically differs from initQ.
        self._update_score_head_label(self.analysisScoreExpLabel, "Score head (initial)", result.get("scoreExp"))
        self._update_score_head_label(self.analysisScoreSearchLabel, "Score head (search)", result.get("scoreSearch"))

        mse = _value_mse(result)
        if mse is not None:
            self.analysisValueMSELabel.configure(text=f"Value MSE loss: {mse:.4f}")

        ce = _policy_visit_cross_entropy(result)
        if ce is not None:
            self.analysisPolicyCELabel.configure(text=f"Policy CE loss: {ce:.4f}")

        status = self._compute_analysis_status(result)
        visits = result.get("visits")
        if visits is not None and status is not None:
            # "This turn"'s visits, not the engine's raw (carryover-inflated) root total --
            # see analysis_turn_base_visits' comment in __init__.
            turn_visits = max(0, int(visits) - self.analysis_turn_base_visits)
            self.analysisVisitsLabel.configure(
                text=f"Visits: {turn_visits} / {self.analysis_turn_requested} ({status})")

        self._update_candidate_listbox(result)
        self._maybe_auto_play(status)

    def _update_score_head_label(self, label, prefix, score):
        if score is None:
            return
        # Turn-relative (same convention as winrate/scoreMap): positive favors whoever is to
        # move, negative favors their opponent. Also spelled out in absolute Black/White terms
        # since that's what the raw net-margin number actually means on the board.
        leader = "Black" if (score > 0) == (self.turn == 0) else "White"
        label.configure(text=f"{prefix}: {score:+.2f} ({leader} by {abs(score):.1f})")

    def _compute_analysis_status(self, result):
        """"paused" / "analyzing..." / "done"[, forced win/loss] -- also doubles as the signal
        _maybe_auto_play watches to know the engine is finished "thinking" about its move.

        A proven win/loss (forced != 0) makes the engine's analyze loop stop for good right
        there, well short of the playout target -- and once the child a move jumps into
        already carries a high visit count from earlier search, the very first request can
        likewise already be at/past the target. Both cases are genuinely done, not stuck;
        without checking them explicitly the label would say "analyzing..." forever even
        though no further update will ever arrive, which looks like a freeze.
        """
        visits = result.get("visits")
        if visits is None:
            return None
        if self.analysis_paused:
            return "paused"
        forced = result.get("forced") or 0
        if forced > 0:
            return "done, forced win"
        if forced < 0:
            return "done, forced loss"
        if visits >= self.analysis_turn_base_visits + self.analysis_turn_requested:
            return "done"
        return "analyzing..."

    def _update_candidate_listbox(self, result):
        show_variation = self.overlay_var.get() == "variation"

        if show_variation:
            self.candidateListHeader.configure(text="Candidate moves, with follow-up line:")
            self.candidateColumnHeaderFrame.pack_forget()
            self.candidateVariationHeader.pack(fill=X, pady=(2, 2))
        else:
            sort_label = {"q": "Q"}.get(self.candidate_sort_key, self.candidate_sort_key)
            self.candidateListHeader.configure(
                text=f"Candidate moves, sorted by {sort_label} "
                     f"({'desc' if self.candidate_sort_reverse else 'asc'}):")
            self.candidateVariationHeader.pack_forget()
            self.candidateColumnHeaderFrame.pack(fill=X, pady=(2, 2))
            arrow = " ▼" if self.candidate_sort_reverse else " ▲"
            for key, lbl in self._candidate_sort_labels.items():
                if key == self.candidate_sort_key:
                    lbl.configure(text=(self._candidate_header_text[key].rstrip() + arrow).ljust(
                        len(self._candidate_header_text[key]) + 2), fg="black")
                else:
                    lbl.configure(text=self._candidate_header_text[key], fg="#555555")

        self.analysisListbox.delete(0, END)

        # Visits are the engine's cumulative total on the current position (every playout run
        # since the last move was played/position changed, not just whatever the most recent
        # search chunk added) -- each candidate row below already reflects that, but the root
        # total itself isn't otherwise shown anywhere in this box, so give it its own row.
        # Row 0 always, and it's not a candidate -- _on_candidate_listbox_select below skips it.
        total_visits = result.get("visits")
        total_text = f"Total visits: {int(total_visits)}" if total_visits is not None else "Total visits: -"
        self.analysisListbox.insert(END, total_text)
        self.analysisListbox.itemconfig(0, fg="#555555")

        moves = _sorted_candidate_moves(result, self.candidate_sort_key, self.candidate_sort_reverse)
        # Once forced, "moves" holds exactly the one proven move (see Analysis::printAnalysis) --
        # already the sole/first row below with no extra sorting needed; itemconfig just flags
        # that row's text color so it reads as "proven", not merely "currently on top".
        forced = result.get("forced") or 0
        selected_row = None
        for idx, mv in enumerate(moves):
            r, c = mv["move"]
            if mv["move"] == self.selected_variation_move:
                selected_row = idx
            if show_variation:
                seq = [(r, c)] + mv.get("variation", [])
                text = " → ".join(f"({sr},{sc})" for sr, sc in seq)
                if forced > 0:
                    text = "★ " + text
            else:
                pct = (mv["winrate"] + 1) / 2 * 100
                text = (f"({r},{c})".ljust(10) + f"{int(mv['visits']):<8}"
                        f"{mv['prior'] * 100:5.1f}%  {pct:5.1f}%  {mv['q']:+.3f}  {mv['pref']:+.3f}")
            row = idx + 1  # offset by the "Total visits" row at index 0
            self.analysisListbox.insert(END, text)
            # Per-move "forced" (see _ANALYSIS_MOVE_RE's comment for the sign convention) covers
            # both cases uniformly: the ordinary per-child breakdown (root itself undetermined,
            # but any individual child can still be a proven win/loss), and the single-move
            # "root already proven" breakdown (whose one move's own forced field was set to match
            # root's clamped result -- see childForced in src/analysis.cpp's forced-root branch).
            mv_forced = mv.get("forced", 0)
            if mv_forced < 0:
                self.analysisListbox.itemconfig(row, fg=FORCED_WIN_COLOR)
            elif mv_forced > 0:
                self.analysisListbox.itemconfig(row, fg=FORCED_LOSS_COLOR)

        if show_variation and selected_row is not None:
            self.analysisListbox.selection_set(selected_row + 1)

    def _on_candidate_listbox_select(self, event=None):
        if self.overlay_var.get() != "variation" or self.last_analysis_result is None:
            return
        sel = self.analysisListbox.curselection()
        if not sel:
            return
        idx = sel[0] - 1  # row 0 is the "Total visits" summary row, not a candidate
        if idx < 0:
            return
        moves = _sorted_candidate_moves(self.last_analysis_result, self.candidate_sort_key,
                                         self.candidate_sort_reverse)
        if idx >= len(moves):
            return
        self.selected_variation_move = moves[idx]["move"]
        self._draw_move_overlay(self.last_analysis_result)

    def _on_candidate_sort_header_click(self, key):
        """Clicking a candidate-list column header sorts by that field; clicking the
        already-active column again just flips descending/ascending, matching the common
        sortable-table convention."""
        if key == self.candidate_sort_key:
            self.candidate_sort_reverse = not self.candidate_sort_reverse
        else:
            self.candidate_sort_key = key
            self.candidate_sort_reverse = True
        if self.last_analysis_result is not None:
            self._update_candidate_listbox(self.last_analysis_result)

    def _on_debug_mode_change(self):
        """Just tells the engine whether to log playouts from here on -- it does not itself
        start or stop search (that's the Pause/Resume button's job, exclusively). So flipping
        this on while search is paused produces nothing until Resume is pressed, and flipping it
        on mid-search starts logging the playouts that search is already running. Either way we
        also reset our own accumulated playout_log locally, since it described whatever was
        captured under the *previous* on/off state, which the freshly (un)toggled engine
        process's stream no longer matches.
        """
        self.playout_log = []
        self.selected_playout_index = None
        on = self.debug_mode_var.get()

        # Both candidateFrame and debugFrame live in their own Search column and stay packed
        # regardless of "on" -- debugFrame just has nothing in it (see _update_debug_listbox)
        # while debug mode is off, rather than being hidden/resized.
        if self.analysisEngine is not None:
            self.analysisEngine.send_debug(on)
        # selected_playout_index was just reset above -- if the board was showing a playout's
        # path (Variation overlay), redraw so that stale marker set doesn't linger.
        if self.last_analysis_result is not None:
            self._draw_move_overlay(self.last_analysis_result)
        self._update_debug_listbox()

    def _update_debug_listbox(self):
        self.debugPathListbox.delete(0, END)
        self.debugListbox.delete(0, END)
        selected_row = None
        for row, rec in enumerate(self.playout_log):
            if row == self.selected_playout_index:
                selected_row = row
            # "row col" per move, no parens/comma (e.g. (2,4) (2,8) -> "24 28") -- much more
            # compact than the tuple form, which is what forced winp/score off-screen for any
            # search path longer than a handful of moves.
            path_text = " ".join(f"{r}{c}" for r, c in rec["path"]) or "(root)"
            if rec["forced"] != 0:
                stat_text = "forced win" if rec["forced"] > 0 else "forced loss"
            else:
                stat_text = f"{rec['winp']:+.2f}  {rec['score']:+.2f}"
            self.debugPathListbox.insert(END, path_text)
            self.debugListbox.insert(END, f"{row:<5} {stat_text}")
        if selected_row is not None:
            self.debugPathListbox.selection_set(selected_row)
            self.debugPathListbox.see(selected_row)
            self.debugListbox.selection_set(selected_row)
            self.debugListbox.see(selected_row)

    def _debug_listboxes_yview(self, *args):
        """Vertical scrollbar command for the split path/stat debug listboxes -- drives both in
        lock-step (see the yscrollcommand sync installed in _build_position_tab)."""
        self.debugPathListbox.yview(*args)
        self.debugListbox.yview(*args)

    def _on_debug_list_mousewheel(self, event):
        if getattr(event, "num", None) == 5 or event.delta < 0:
            units = 1
        else:
            units = -1
        self.debugPathListbox.yview_scroll(units, "units")
        self.debugListbox.yview_scroll(units, "units")
        return "break"

    def _on_debug_listbox_select(self, event=None):
        # Either listbox can fire this (clicking a row in "path" or in "# winp score") -- mirror
        # the selection onto the other one so they always highlight the same row.
        source = event.widget if event is not None else self.debugPathListbox
        sel = source.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.playout_log):
            return
        other = self.debugListbox if source is self.debugPathListbox else self.debugPathListbox
        other.selection_clear(0, END)
        other.selection_set(idx)
        self.selected_playout_index = idx
        # Selecting a path always traces it on the board -- the same numbered-marker drawing
        # the "Variation" board-overlay mode uses (_draw_playout_path_overlay), so just switch
        # into that mode rather than requiring it be picked first; clicking an "expected
        # continuation" candidate works the same way once already in Variation mode (see
        # _on_candidate_listbox_select) -- this makes debug-path clicks equally immediate.
        self.overlay_var.set("variation")
        if self.last_analysis_result is not None:
            self._draw_move_overlay(self.last_analysis_result)

    def _on_engine_plays_change(self):
        """Called when either "Engine plays" checkbox is toggled. The color assignment can
        change at any moment -- including right after the engine already finished thinking
        about the position on the board -- so re-check immediately instead of waiting for
        the next streamed result, which may not come until something re-requests analysis.
        """
        self._update_status()
        if self.last_analysis_result is not None:
            self._maybe_auto_play(self._compute_analysis_status(self.last_analysis_result))

    def _maybe_auto_play(self, status):
        """Once the engine is done "thinking" (status is a "done..." status) about a position
        whose turn it's been told to auto-play, pick and commit its move. Guarded by
        _auto_move_seq_len so this fires exactly once per position -- several streamed
        results can report "done" for the same position (this callback re-runs on ordinary
        checkbox toggles too), and _commit_move's own re-request would otherwise reopen the
        window for a second, stale trigger.
        """
        if status is None or not status.startswith("done"):
            return
        if self.game_over or not self._is_engine_turn():
            return
        seq_len = len(self.rule.seq)
        if self._auto_move_seq_len == seq_len:
            return
        self._auto_move_seq_len = seq_len
        self.root.after(250, self._auto_play_move)

    def _auto_play_move(self):
        """Commits the engine's chosen move: the current candidate with the most search
        visits, same top-of-the-list move already shown in the candidate panel (a forced
        win/loss position has exactly one candidate -- the proven winning/best-delaying
        move surfaced by Analysis::printAnalysis, so the same "most visits" pick applies there
        too). Re-checks everything since this runs after an after() delay, by which time the
        checkbox could have been unticked or the position could have moved on already.
        """
        if self.game_over or not self.analysis_mode or not self._is_engine_turn():
            return
        result = self.last_analysis_result
        if not result or not result.get("moves"):
            return
        if len(self.rule.seq) != self._auto_move_seq_len:
            return
        best = max(result["moves"], key=lambda mv: mv["visits"])
        if best["visits"] <= 0:
            return
        self._commit_move(*best["move"])

    def _open_match_dialog(self):
        """Engine-vs-engine match dialog: two independently configured engines (model, playout
        budget, move-selection temperature -- see MatchRunner) play a requested number of games
        against each other, alternating colors, with every move/evaluation/result recorded and
        saved to a JSON file under matches/. Fully separate from the interactive board/session
        above -- the match runs its own AnalysisEngine processes on a background thread and
        never touches self.analysisEngine or self.rule.
        """
        models = list_models()
        if not models:
            return

        dialog = Toplevel(self.root)
        dialog.title("Engine vs Engine Match")
        dialog.transient(self.root)
        dialog.runner = None
        dialog.records = []
        dialog.tally = {"A": 0, "B": 0, "draw": 0}
        dialog.match_filename = None
        dialog.match_done = True

        config_frame = Frame(dialog)
        config_frame.pack(padx=10, pady=10, fill=X)

        def build_engine_column(parent, label):
            col = Frame(parent, padx=10)
            Label(col, text=label, font=("Helvetica", 11, "bold")).pack(anchor="w")
            model_var = StringVar(value=models[0])
            OptionMenu(col, model_var, *models).pack(fill=X, pady=(2, 6))
            Label(col, text="Playouts:", anchor="w").pack(fill=X)
            playouts_var = StringVar(value="800")
            Entry(col, textvariable=playouts_var, width=10).pack(anchor="w", pady=(0, 6))
            Label(col, text="Temperature:", anchor="w").pack(fill=X)
            temp_var = StringVar(value="1.0")
            Entry(col, textvariable=temp_var, width=10).pack(anchor="w")
            return col, model_var, playouts_var, temp_var

        col_a, dialog.model_a, dialog.playouts_a, dialog.temp_a = build_engine_column(config_frame, "Engine A")
        col_a.pack(side=LEFT)
        col_b, dialog.model_b, dialog.playouts_b, dialog.temp_b = build_engine_column(config_frame, "Engine B")
        col_b.pack(side=LEFT)

        Label(dialog, text="(Temperature: low = more random move choice, >= 5 = always the "
                            "most-visited move; same convention as MCTS::getMove.)",
              wraplength=420, justify=LEFT, fg="#555555").pack(padx=10, anchor="w")

        games_row = Frame(dialog)
        games_row.pack(padx=10, pady=(8, 0), fill=X)
        Label(games_row, text="Number of games:").pack(side=LEFT)
        dialog.n_games_var = StringVar(value="10")
        Entry(games_row, textvariable=dialog.n_games_var, width=6).pack(side=LEFT, padx=(5, 0))

        btn_row = Frame(dialog)
        btn_row.pack(padx=10, pady=10, fill=X)
        dialog.start_button = Button(btn_row, text="Start Match", command=lambda: self._start_match(dialog))
        dialog.start_button.pack(side=LEFT)
        dialog.stop_button = Button(btn_row, text="Stop", state=DISABLED, command=lambda: self._stop_match(dialog))
        dialog.stop_button.pack(side=LEFT, padx=(5, 0))
        Button(btn_row, text="Close", command=lambda: self._close_match_dialog(dialog)).pack(side=RIGHT)

        dialog.status_label = Label(dialog, text="Configure both engines and press Start.",
                                     anchor="w", wraplength=420, justify=LEFT)
        dialog.status_label.pack(padx=10, fill=X)
        dialog.tally_label = Label(dialog, text="", anchor="w")
        dialog.tally_label.pack(padx=10, fill=X)

        list_frame = Frame(dialog)
        list_frame.pack(padx=10, pady=(5, 10), fill=BOTH, expand=True)
        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        dialog.games_listbox = Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        font=("Courier", 9), width=60, height=12, exportselection=False)
        scrollbar.configure(command=dialog.games_listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        dialog.games_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        dialog.games_listbox.bind("<Double-Button-1>", lambda e: self._load_match_game_from_dialog(dialog))

        Label(dialog, text="Double-click a finished game to load it onto the main board for review.",
              fg="#555555", anchor="w").pack(padx=10, pady=(0, 10), fill=X)

        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_match_dialog(dialog))

    def _start_match(self, dialog):
        try:
            playouts_a = max(1, int(dialog.playouts_a.get()))
            playouts_b = max(1, int(dialog.playouts_b.get()))
            temp_a = max(0.0, float(dialog.temp_a.get()))
            temp_b = max(0.0, float(dialog.temp_b.get()))
            n_games = max(1, int(dialog.n_games_var.get()))
        except ValueError:
            dialog.status_label.configure(text="Playouts/temperature/games must be numbers.")
            return

        cfg_a = {"model": dialog.model_a.get(), "playouts": playouts_a, "temp": temp_a}
        cfg_b = {"model": dialog.model_b.get(), "playouts": playouts_b, "temp": temp_b}

        dialog.records = []
        dialog.tally = {"A": 0, "B": 0, "draw": 0}
        dialog.games_listbox.delete(0, END)
        dialog.cfg_a, dialog.cfg_b, dialog.n_games = cfg_a, cfg_b, n_games

        os.makedirs(MATCHES_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        a_tag = os.path.splitext(cfg_a["model"])[0]
        b_tag = os.path.splitext(cfg_b["model"])[0]
        dialog.match_filename = os.path.join(MATCHES_DIR, f"match_{a_tag}_vs_{b_tag}_{stamp}.json")

        dialog.start_button.configure(state=DISABLED)
        dialog.stop_button.configure(state=NORMAL)
        dialog.status_label.configure(text=f"Starting match -- {n_games} game(s)...")

        dialog.runner = MatchRunner(cfg_a, cfg_b, n_games, build_dir=BUILD_DIR)
        dialog.match_done = False
        dialog.runner.start()
        self.root.after(150, self._poll_match, dialog)

    def _stop_match(self, dialog):
        if dialog.runner is not None:
            dialog.runner.stop()
            dialog.status_label.configure(text="Stopping after the current move...")

    def _close_match_dialog(self, dialog):
        if dialog.runner is not None:
            dialog.runner.stop()
        dialog.destroy()

    @staticmethod
    def _match_row_text(rec):
        black, white = rec["black"], rec["white"]
        if rec["winner"] == "draw":
            res = "draw"
        elif rec["winner"] is None:
            res = f"unresolved ({rec['by']})"
        else:
            loser = white if rec["winner"] == black else black
            margin_txt = f" by {rec['margin']:.1f}" if rec.get("margin") else ""
            res = f"{rec['winner']} beat {loser}{margin_txt} ({rec['by']})"
        return f"Game {rec['index'] + 1:>3}: {black}(black) vs {white}(white) -- {res}"

    @staticmethod
    def _resolve_match_names(rec, model_a, model_b):
        """Returns a shallow copy of a MatchRunner-produced game record with the internal "A"/
        "B" engine tags -- in black/white, winner, and each move's "by" -- resolved to the
        actual model filenames, for display purposes only. The original record (what MatchRunner
        itself produces, what the live match dialog's tally keys off of, and what
        _save_match_records writes to disk) is left untouched; only this copy, handed to
        _match_row_text/_review_row_text/_update_moves_tab, is touched. Passing None for
        either name leaves that tag as-is (e.g. a match file saved before this existed and
        missing engineA/engineB "model").
        """
        names = {"A": model_a or "A", "B": model_b or "B"}
        out = dict(rec)
        out["black"] = names.get(rec.get("black"), rec.get("black"))
        out["white"] = names.get(rec.get("white"), rec.get("white"))
        out["winner"] = names.get(rec.get("winner"), rec.get("winner"))
        out["moves"] = [dict(mv, by=names.get(mv.get("by"), mv.get("by"))) if "by" in mv else mv
                         for mv in rec.get("moves", [])]
        return out

    @staticmethod
    def _display_side_label(tag):
        """Maps a stored mover/side tag to what's actually shown to the user: "human" (the
        internal tag _commit_move/_save_session_game write -- see session_move_log) reads as
        "player" everywhere in the UI instead. Everything else -- a model filename, "engine",
        "mixed", or an "A"/"B" match tag already resolved by _resolve_match_names -- passes
        through unchanged. The stored value itself stays "human"; only display goes through this.
        """
        return "player" if tag == "human" else tag

    @staticmethod
    def _review_row_text(rec):
        """One-line result summary for a record shown in review mode -- unlike _match_row_text
        above (which is specific to the live match dialog's listbox), this only relies on
        winner_color/margin/by/black/white, fields both MatchRunner's match records and
        _save_session_game's auto-saved session records carry, so it works uniformly on either
        (record["winner"] itself isn't: it's an engine tag ("A"/"B") on a match record but a
        color ("black"/"white") on a session record -- winner_color sidesteps that entirely).
        """
        black = Game._display_side_label(rec.get("black", "?"))
        white = Game._display_side_label(rec.get("white", "?"))
        wc = rec.get("winner_color")
        by = rec.get("by")
        if wc == 1:
            res = f"black ({black}) wins ({by})"
        elif wc == -1:
            res = f"white ({white}) wins ({by})"
        elif wc == 0:
            res = f"draw ({by})"
        else:
            res = f"unresolved ({by})"
        margin = rec.get("margin")
        if margin:
            res += f", by {margin:.1f}"
        return f"black: {black}, white: {white} -- {res}"

    def _poll_match(self, dialog):
        if not dialog.winfo_exists():
            return  # dialog was closed; _close_match_dialog already stopped the runner

        while True:
            try:
                event = dialog.runner.events.get_nowait()
            except queue.Empty:
                break

            if event["type"] == "game_start":
                dialog.status_label.configure(
                    text=f"Game {event['index'] + 1}/{dialog.n_games} running "
                         f"({event['black']} black, {event['white']} white)...")
            elif event["type"] == "game_end":
                rec = event["record"]
                dialog.records.append(rec)
                if rec["winner"] in dialog.tally:
                    dialog.tally[rec["winner"]] += 1
                display_rec = self._resolve_match_names(rec, dialog.cfg_a["model"], dialog.cfg_b["model"])
                dialog.games_listbox.insert(END, self._match_row_text(display_rec))
                dialog.games_listbox.see(END)
                dialog.tally_label.configure(
                    text=f"A wins: {dialog.tally['A']}   B wins: {dialog.tally['B']}   "
                         f"draws: {dialog.tally['draw']}")
                self._save_match_records(dialog)
            elif event["type"] == "done":
                dialog.match_done = True
                dialog.start_button.configure(state=NORMAL)
                dialog.stop_button.configure(state=DISABLED)
                finished = len(dialog.records)
                saved = f" Saved to {dialog.match_filename}." if dialog.match_filename else ""
                dialog.status_label.configure(text=f"Match finished -- {finished}/{dialog.n_games} game(s).{saved}")

        if not dialog.match_done:
            self.root.after(150, self._poll_match, dialog)

    def _save_match_records(self, dialog):
        data = {
            "engineA": dialog.cfg_a,
            "engineB": dialog.cfg_b,
            "n_games_requested": dialog.n_games,
            "tally": dialog.tally,
            "games": dialog.records,
        }
        with open(dialog.match_filename, "w") as f:
            json.dump(data, f, indent=2)

    def _load_match_game_from_dialog(self, dialog):
        sel = dialog.games_listbox.curselection()
        if not sel or sel[0] >= len(dialog.records):
            return
        rec = dialog.records[sel[0]]
        display_rec = self._resolve_match_names(rec, dialog.cfg_a["model"], dialog.cfg_b["model"])
        self._load_game_for_reference(display_rec)

    def _open_load_game_dialog(self):
        """Entry point for loading any previously recorded game -- an auto-saved human/engine
        session (see _save_session_game) or one game out of a saved engine-match file (see
        MatchRunner) -- as the Moves tab's reference data (_load_game_for_reference). Reachable
        both from the setup panel (nothing running yet) and from the Moves tab itself (to swap
        in a different game mid-session). Distinct from the live match dialog's own
        double-click-to-load (_load_match_game_from_dialog): this one reads back off disk, so it
        also reaches games from past sessions of the program, not just the current one.
        """
        entries = self._scan_matches_dir()

        dialog = Toplevel(self.root)
        dialog.title("Load Game")
        dialog.transient(self.root)

        Label(dialog, text="Games found under matches/ (newest first):",
              font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        list_frame = Frame(dialog)
        list_frame.pack(fill=BOTH, expand=True, padx=10)
        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Courier", 9),
                           width=72, height=18, exportselection=False)
        scrollbar.configure(command=listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        listbox.pack(side=LEFT, fill=BOTH, expand=True)

        if not entries:
            listbox.insert(END, "(no games found under matches/)")
            listbox.configure(state=DISABLED)
        else:
            for text, _record, _mtime in entries:
                listbox.insert(END, text)

        def load(event=None):
            sel = listbox.curselection()
            if not sel or sel[0] >= len(entries):
                return
            record = entries[sel[0]][1]
            dialog.destroy()
            self._load_game_for_reference(record)

        listbox.bind("<Double-Button-1>", load)

        btn_row = Frame(dialog)
        btn_row.pack(fill=X, padx=10, pady=10)
        Button(btn_row, text="Load", command=load).pack(side=LEFT)
        Button(btn_row, text="Close", command=dialog.destroy).pack(side=RIGHT)

    def _scan_matches_dir(self):
        """Every individually-loadable game found under matches/: match files (MatchRunner's
        output -- one file, many games) are expanded to one entry per game; auto-saved session
        files (_save_session_game -- one game per file) are one entry each. Returns a list of
        (display_text, record, mtime) tuples, newest file first."""
        entries = []
        if not os.path.isdir(MATCHES_DIR):
            return entries
        for fname in sorted(os.listdir(MATCHES_DIR)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(MATCHES_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            mtime = os.path.getmtime(path)
            if "games" in data:
                model_a = data.get("engineA", {}).get("model")
                model_b = data.get("engineB", {}).get("model")
                for rec in data["games"]:
                    display_rec = self._resolve_match_names(rec, model_a, model_b)
                    entries.append((f"{fname} -- {self._match_row_text(display_rec)}", display_rec, mtime))
            elif "moves" in data:
                entries.append((f"{fname} -- {self._review_row_text(data)}", data, mtime))
        entries.sort(key=lambda e: e[2], reverse=True)
        return entries

    def _load_game_for_reference(self, record):
        """Loads a recorded game as the Moves tab's reference data (its B eval/W eval columns --
        see _update_moves_tab) and jumps the board to its final position. review_mode locks out
        board clicks/pass while a game is loaded (see on_click/on_pass/_is_engine_turn) -- it's
        fixed, already-finished history to browse and analyze, not something to play new moves
        into. If a live analysis session is already running, it's kept (and resynced to the new
        position by _jump_to_ply below) rather than closed, so swapping which game is being
        browsed mid-session doesn't interrupt analysis; if none is running yet, the board/rule
        are simply rebuilt and the B/W columns show up immediately, ready for "Start Analyzing"
        (with any model) whenever you like.
        """
        self.engine_plays_black_var.set(False)
        self.engine_plays_white_var.set(False)
        self.position_eval_log = {}

        self.review_mode = True
        self.review_record = record
        self.review_moves = [mv["move"] for mv in record.get("moves", [])]

        self._show_panel(self.analysisResultPanel)
        self._jump_to_ply(len(self.review_moves))

    def _start_analysis_from_panel(self):
        """"Start Analyzing" in the Moves tab: attaches a live engine (self.setup_model_var,
        shared with the setup panel's own model picker) at whatever position is currently on the
        board -- typically reached right after loading a game for reference
        (_load_game_for_reference) and jumping to whichever of its positions is of interest, but
        works the same with no game loaded too (equivalent to the setup panel's own "Start
        Session", just reachable from inside an already-open Analysis window).
        """
        model = self.setup_model_var.get()
        if not model:
            return
        self.start_session(model)

    def _current_moves_list(self):
        """The move-record list the Moves tab/board navigation currently browse: a loaded game's
        recorded moves (self.review_record -- fixed, read-only) if one is loaded, else this
        session's own move history (self.session_move_log -- grows as moves are played, and can
        be rewound/diverged from, see _commit_move's truncation). Each entry has at least "ply"
        and "move"; loaded-game entries additionally carry "root_winrate" (that game's own
        recorded evaluation -- see _update_moves_tab's B eval/W eval column).
        """
        if self.review_record is not None:
            return self.review_record.get("moves", [])
        return self.session_move_log

    def _current_moves_coords(self):
        return [mv["move"] for mv in self._current_moves_list()]

    def _jump_to_ply(self, idx):
        """Moves the board to the position after `idx` moves of whichever move list is currently
        being browsed (_current_moves_coords) -- rebuilding self.rule/the board from scratch
        (_render_review_position) rather than trying to incrementally undo capture/territory
        bookkeeping (simple, and at this board size/move count cheap enough not to matter) -- and,
        if a live analysis session is attached, resyncing the engine to the same position
        (_sync_analysis_position, the same reset-and-replay it already does on session start)
        and immediately requesting fresh analysis there, so re-visiting a former position during
        live analysis is how its "Current evaluation" (position_eval_log, updated by
        _update_analysis_result_panel) gets (re-)filled in.
        """
        moves = self._current_moves_coords()
        idx = max(0, min(idx, len(moves)))
        self._render_review_position(idx, moves)

        if self.analysis_mode and self.analysisEngine is not None:
            self.last_analysis_result = None
            self.canvas.delete("move_overlay")
            self._sync_analysis_position()
            self.apply_analysis_limit()

        self._update_moves_tab()

    def _render_review_position(self, idx, moves=None):
        if moves is None:
            moves = self._current_moves_coords()
        self.review_index = idx
        self.rule = RuleManager()
        self.canvas.delete("stone")
        self.canvas.delete("move_overlay")
        self.on_stone = [[False for _ in range(self.boardSize)] for _ in range(self.boardSize)]
        self.turn = 0
        for mv in moves[:idx]:
            self._apply_review_move(mv)
        self._mark_last_review_move(moves)

        if self.review_record is not None:
            if idx == len(moves):
                self.statusLabel.configure(
                    text=f"Reviewing final position -- {self._review_row_text(self.review_record)}")
            else:
                mover = "Black" if self.turn == 0 else "White"
                self.statusLabel.configure(text=f"Reviewing move {idx} / {len(moves)} -- {mover} to move next")
        else:
            # Free play (no game loaded): game_over only reflects reality at the exact ply the
            # actual game ended on (see on_end) -- jumping away from it means the game isn't
            # over from here, jumping back onto it restores the exact "who won and how" text.
            self.game_over = (self.game_over_ply is not None and idx == self.game_over_ply)
            if self.game_over:
                self.statusLabel.configure(text=self.game_over_text or "")
            elif self.analysis_mode:
                self._update_status()
            else:
                self.statusLabel.configure(text="Black to move" if self.turn == 0 else "White to move")

    def _apply_review_move(self, mv):
        """Replays one recorded move onto self.rule/the board, mirroring exactly what
        _commit_move does for a live move (pass-vs-stone turn bookkeeping included) but with
        none of the analysis-engine/logging/game-over side effects -- _render_review_position
        already knows (or doesn't need) the final result and just wants the board/rule state
        rebuilt to match the position right after mv."""
        if mv is None:
            result = self.rule.make_move(-1, 0)
            if result != -2:
                self.turn = 1 - self.turn
        else:
            x, y = mv
            self._place_stone(x, y, self.turn)
            self.turn = 1 - self.turn
            self.rule.make_move(x, y)

    def _mark_last_review_move(self, moves):
        """Small marker on the most recently replayed stone, so it's obvious at a glance which
        move a jump just landed on -- skipped for a pass, which has no board point to mark."""
        if self.review_index == 0:
            return
        mv = moves[self.review_index - 1]
        if mv is None:
            return
        x, y = mv
        cx, cy = self.m + x * self.delta, self.m + y * self.delta
        r = max(3, int(self.stone_size * 0.12))
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#e53935", outline="",
                                 tags=("move_overlay", "move_overlay_shape"))

    @staticmethod
    def _fmt_winrate_pct(wr):
        return f"{(wr + 1) / 2 * 100:.0f}%" if wr is not None else ""

    def _update_moves_tab(self):
        """Refreshes the Moves tab: the info line (which game, if any, is loaded, and its
        result), the move table, and the position counter -- see _current_moves_list for which
        move source is being shown. Per the "B eval"/"W eval"/"Current evaluation" convention:
        B/W come from a *loaded* game's own recorded per-ply evaluation (root_winrate -- turn-
        relative, so it's already "how did this look for the side about to move," i.e. self-
        relative for whichever color's row it is) and stay blank with no game loaded, since
        there's no separate "recorded by the original engine" evaluation to show then; Current
        always comes from position_eval_log -- the position-currently-being-analyzed engine's
        own evaluation of that same ply, filled in as it's (re-)visited during live analysis,
        regardless of whether a game is loaded.
        """
        if self.review_record is not None:
            info = self._review_row_text(self.review_record)
            model = self.review_record.get("model")
            if model:
                info += f"\nModel: {model}"
        else:
            info = "No game loaded -- showing this session's own move history."
        self.movesInfoLabel.configure(text=info)

        moves = self._current_moves_list()
        for row in self.movesTree.get_children():
            self.movesTree.delete(row)
        selected_iid = None
        for i, mv in enumerate(moves):
            ply = mv.get("ply", i)
            color = "black" if ply % 2 == 0 else "white"
            move = mv.get("move")
            move_txt = "pass" if move is None else f"({move[0]},{move[1]})"

            b_eval = w_eval = ""
            if self.review_record is not None:
                txt = self._fmt_winrate_pct(mv.get("root_winrate"))
                if color == "black":
                    b_eval = txt
                else:
                    w_eval = txt
            cur_eval = self._fmt_winrate_pct(self.position_eval_log.get(ply, {}).get("winrate"))

            iid = str(i)
            self.movesTree.insert("", END, iid=iid, values=(ply + 1, move_txt, b_eval, w_eval, cur_eval))
            if self.review_index == i + 1:
                selected_iid = iid

        # No explicit deselect needed for the "no current ply" case: every row above was just
        # deleted and freshly reinserted, so there's nothing left selected unless set_selection
        # was just called for it.
        if selected_iid is not None:
            self.movesTree.selection_set(selected_iid)
            self.movesTree.see(selected_iid)

        self.movesPositionLabel.configure(text=f"Position {self.review_index} / {len(moves)}")

    def _on_moves_tree_select(self, event=None):
        sel = self.movesTree.selection()
        if not sel:
            return
        idx = int(sel[0]) + 1
        if idx == self.review_index:
            return  # already there -- avoid re-jumping on the selection _update_moves_tab itself sets
        self._jump_to_ply(idx)

    def on_end(self, winner):
        self.game_over = True
        self.game_over_ply = len(self.rule.seq)
        if winner == 1:
            winner_color, margin, by = 1, None, "capture"
            self.statusLabel.configure(text="winner is black")
        elif winner == -1:
            winner_color, margin, by = -1, None, "capture"
            self.statusLabel.configure(text="winner is white")
        else:
            result = self.rule.end_game()
            winner_color, margin, by = result[0], (result[1] or None), "score"
            if result[0] == 1:
                self.statusLabel.configure(text=f"winner is black by {result[1]}")
            elif result[0] == -1:
                self.statusLabel.configure(text=f"winner is white by {result[1]}")
            else:
                self.statusLabel.configure(text="draw")
        # Restored verbatim by _render_review_position if a later free-play jump (_jump_to_ply)
        # lands back on game_over_ply -- game_over alone doesn't carry the "who won and how" text.
        self.game_over_text = self.statusLabel.cget("text")

        self._save_session_game(winner_color, margin, by)

        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None

    def _save_session_game(self, winner_color, margin, by):
        """Auto-saves a finished game that had a live analysis session at some point -- human vs
        engine, engine vs engine played interactively, or human vs human under analysis (see
        session_used_engine, set by start_session) -- to matches/, in the same per-move record
        shape MatchRunner's match files use (see MatchRunner._run), so both the engine-match
        dialog's loader and _open_load_game_dialog/_load_game_for_reference work on either source
        uniformly. A pure "New Local Game" (no session ever started) isn't saved -- there's no
        evaluation data worth keeping, and that mode is meant to stay a throwaway scratch board.
        """
        if not self.session_used_engine or not self.session_move_log:
            return

        def side_desc(color):
            movers = {rec["mover"] for rec in self.session_move_log if rec["color"] == color}
            if not movers:
                return "human"
            return next(iter(movers)) if len(movers) == 1 else "mixed"

        black, white = side_desc("black"), side_desc("white")
        winner = {1: "black", -1: "white", 0: "draw"}.get(winner_color)

        record = {
            "type": "session",
            "model": self.session_model,
            "black": black, "white": white,
            "winner": winner, "winner_color": winner_color,
            "margin": margin, "by": by,
            "moves": self.session_move_log,
        }

        os.makedirs(MATCHES_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = os.path.splitext(self.session_model)[0] if self.session_model else "session"
        path = os.path.join(MATCHES_DIR, f"game_{tag}_{black}_vs_{white}_{stamp}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def on_close(self):
        if self.analysisEngine is not None:
            self.analysisEngine.close()
        self.root.destroy()

    @staticmethod
    def start_self_play(player, is_shown=False, temp=1e-1):
        rule_manager = RuleManager()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(rule_manager, temp=temp, return_prob=True)
            states.append(rule_manager.current_state())
            mcts_probs.append(move_probs)
            current_players.append(rule_manager.turn)
            result = rule_manager.make_move(move)
            # if is_shown:
            #     continue
            if result != 0:
                winners_z = np.zeros(len(current_players), dtype=float)
                if result != -2:
                    result *= rule_manager.turn
                else:
                    result = rule_manager.end_game()[0]
                winners_z[np.array(current_players) == result] = 1
                winners_z[np.array(current_players) != result] = -1
                player.reset_player()
                if is_shown:
                    print("winner :", result)
                return result, zip(states, mcts_probs, winners_z)

    @staticmethod
    def start_play(player1, player2, start_player=0, is_shown=False, temp=1e-1):
        rule_manager = RuleManager()
        t = [-1, 0, 1]
        if start_player == 0:
            player_list = [player1, player2]
        else:
            player_list = [player2, player1]

        while True:
            current_player = player_list[t[rule_manager.turn]]
            move = current_player.get_action(rule_manager, temp=temp)
            res = rule_manager.make_move(move)

            if res == 0:
                continue
            if is_shown:
                print(rule_manager.seq)
            if res == -2:
                res = rule_manager.end_game()
                if res == 0:
                    print("draw")
                    return 0.5
                else:
                    if res[0] == 1 and start_player == 0:
                        print("winner is ", "current player", "win by point")
                        return 1
                    if res[0] == -1 and start_player == 1:
                        print("winner is ", "current player", "win by point")
                        return 1
                    if res[0] == 0:
                        print("result is draw")
                        return 0.5
                    print("winner is ", "opponent ", "win by point")
                    return 0
            if res * rule_manager.turn == 1 and start_player == 0:
                print("winner is ", "current player", "win by capture")
                return 1
            if res * rule_manager.turn == -1 and start_player == 1:
                print("winner is ", "current player", "win by capture")
                return 1

            print("winner is ", "opponent ", "win by capture")
            return 0


# RuleManager.boardSize = 3
# RuleManager.neutral = []
# RuleManager.penalty = 0
if __name__ == "__main__":
    g = Game()
    g.start()
    g.root.mainloop()
