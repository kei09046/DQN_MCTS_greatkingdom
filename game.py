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

from tkinter import *
from PIL import Image, ImageTk

# 실제 게임이 실행되는 파일 GUI 담당이기도 함

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

BOARD_BG = "#DCB35C"
OVERLAY_COLOR = "#1565C0"  # blended toward this color for stronger policy/visit intensity

# A Node only gets its own per-move breakdown (edgeP/edgeN -- see MCTS::printAnalysis in
# src/PMCTS.cpp) once *something* has searched through it as an internal node, not merely
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


def _sorted_candidate_moves(result):
    """Candidate moves with at least one visit, sorted by visit count -- the exact set/order the
    candidate Listbox is populated in, so a row index maps back to the same move on either side."""
    moves = sorted(result.get("moves", []), key=lambda m: m["visits"], reverse=True)
    return [mv for mv in moves if mv["visits"] > 0]


_FLOAT = r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"

# ---- ModelCompare::analyze() protocol (see MCTS::printAnalysis in src/PMCTS.cpp) ----
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
# "move <r> <c> visits <N> prior <P> winrate <W> q <Q> variation <r1> <c1> <r2> <c2> ..." --
# one candidate move; the variation tail is 0 or more coordinate pairs, captured whole and
# split separately since its length varies per move.
_ANALYSIS_MOVE_RE = re.compile(
    r"^move\s+(-?\d+)\s+(-?\d+)\s+visits\s+" + _FLOAT + r"\s+prior\s+" + _FLOAT
    + r"\s+winrate\s+" + _FLOAT + r"\s+q\s+" + _FLOAT + r"\s+variation(.*)$")
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
# while debug mode is on (see MCTS::setDebugMode/printPlayoutDebugLine in src/PMCTS.h/.cpp).
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
        #                 "moves": [{"move": (r,c), "visits": float, "prior": float, "winrate": float}, ...]}
        self.results = queue.Queue()
        # playouts: individual debug-mode playout dicts {"forced": int, "winp": float,
        # "score": float, "path": [(r,c), ...]}, one per line, streamed live and independent of
        # the "analysis begin"/"analysis end" block cycle -- the engine doesn't accumulate these
        # itself (see printPlayoutDebugLine in src/PMCTS.cpp), so building up the growing list a
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
                r, c, visits, prior, winrate, q, var_tail = m.groups()
                var_ints = [int(x) for x in var_tail.split()]
                variation = list(zip(var_ints[0::2], var_ints[1::2]))
                current["moves"].append({
                    "move": (int(r), int(c)),
                    "visits": float(visits),
                    "prior": float(prior),
                    "winrate": float(winrate),
                    "q": float(q),
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
        self.root.geometry("1420x1000")

        self.boardFrame = Frame(self.root)
        self.canvas = Canvas(self.boardFrame, width=900, height=900, bg=BOARD_BG, highlightthickness=0)
        self.passButton = Button(self.boardFrame, text="pass")
        self.statusLabel = Label(self.boardFrame, text="", font=("Helvetica", 14))

        self.sideFrame = Frame(self.root, width=420, height=900)
        self.sideFrame.pack_propagate(False)
        self.setupPanel = None
        self.analysisResultPanel = None
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
        panel = Frame(self.sideFrame)
        self.analysisResultPanel = panel

        Label(panel, text="Session", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 15))

        Label(panel, text="Engine plays:", anchor="w").pack(fill=X)
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

        # candidateFrame (ordinary per-move breakdown) and debugFrame (per-playout log) occupy
        # the same spot and are never both visible -- _on_debug_mode_change toggles between them.
        list_area = Frame(panel)
        list_area.pack(fill=BOTH, expand=True)

        self.candidateFrame = Frame(list_area)
        self.candidateListHeader = Label(self.candidateFrame, text="Candidate moves, by visit count:",
                                          font=("Helvetica", 10, "bold"), anchor="w",
                                          wraplength=260, justify=LEFT)
        self.candidateListHeader.pack(fill=X)
        self.candidateColumnHeader = Label(self.candidateFrame, text="move       visits   policy  winrate     Q",
                                            font=("Courier", 9), fg="#555555", anchor="w")
        self.candidateColumnHeader.pack(fill=X, pady=(2, 2))

        candidate_list_frame = Frame(self.candidateFrame)
        candidate_list_frame.pack(fill=BOTH, expand=True)
        candidate_scrollbar = Scrollbar(candidate_list_frame, orient=VERTICAL)
        self.analysisListbox = Listbox(candidate_list_frame, yscrollcommand=candidate_scrollbar.set,
                                        font=("Courier", 10), height=20, exportselection=False)
        candidate_scrollbar.configure(command=self.analysisListbox.yview)
        candidate_scrollbar.pack(side=RIGHT, fill=Y)
        self.analysisListbox.pack(side=LEFT, fill=BOTH, expand=True)
        self.analysisListbox.bind("<<ListboxSelect>>", self._on_candidate_listbox_select)
        self.candidateFrame.pack(fill=BOTH, expand=True)

        self.debugFrame = Frame(list_area)
        Label(self.debugFrame, text="Playouts, in search order -- select one to trace its\n"
                                     "path on the board (Variation overlay mode):",
              font=("Helvetica", 10, "bold"), anchor="w", wraplength=260, justify=LEFT).pack(fill=X)
        Label(self.debugFrame, text="#     path                              winp   score",
              font=("Courier", 9), fg="#555555", anchor="w").pack(fill=X, pady=(2, 2))

        debug_list_frame = Frame(self.debugFrame)
        debug_list_frame.pack(fill=BOTH, expand=True)
        debug_scrollbar = Scrollbar(debug_list_frame, orient=VERTICAL)
        self.debugListbox = Listbox(debug_list_frame, yscrollcommand=debug_scrollbar.set,
                                     font=("Courier", 9), height=20, exportselection=False)
        debug_scrollbar.configure(command=self.debugListbox.yview)
        debug_scrollbar.pack(side=RIGHT, fill=Y)
        self.debugListbox.pack(side=LEFT, fill=BOTH, expand=True)
        self.debugListbox.bind("<<ListboxSelect>>", self._on_debug_listbox_select)
        # debugFrame starts unpacked -- candidateFrame is the default view.

        Button(panel, text="End Session / Back to Setup", command=self.end_analysis_mode).pack(fill=X, pady=(10, 0))

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
        playing a side, the human playing it, and pure hands-on analysis of both sides."""
        if not self.analysis_mode:
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
        """
        is_pass = (x == self.boardSize and y == 0)
        if is_pass:
            result = self.rule.make_move(-1, 0)
        else:
            self._place_stone(x, y, self.turn)
            self.turn = 1 - self.turn
            result = self.rule.make_move(x, y)

        if self.analysis_mode:
            self.analysisEngine.send_play(x, y)
            self.analysisEngine.request_analysis(
                _PAUSED_MOVE_MIN_VISITS if self.analysis_paused else self._get_analysis_limit())
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
        if self.game_over:
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
        if self.game_over:
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
        self.debugListbox.delete(0, END)
        self.debugFrame.pack_forget()
        self.candidateFrame.pack(fill=BOTH, expand=True)

    def reset_board(self):
        """Clear the board, end any live session, and start a fresh local (no engine) game."""
        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None
        self.canvas.delete("stone")
        self.canvas.delete("move_overlay")
        self.last_analysis_result = None
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
        self.statusLabel.configure(text="")
        self._show_panel(self.setupPanel)
        self._update_status()

    def start_session(self, model=None):
        """Launch (or restart) the analysis engine on the position currently on the board,
        and set which side(s) -- if any -- it auto-plays for, per the setup panel's choice.
        Since this doesn't clear the board first, it doubles as both "analyze this position"
        and "play a game against the engine from here"; which one it feels like is entirely
        down to the Engine plays checkboxes, freely changeable afterwards from the session
        panel without ending the session.
        """
        if model is None:
            model = self.setup_model_var.get()
        if not model:
            return

        if self.analysisEngine is not None:
            self.analysisEngine.close()

        self.analysisEngine = AnalysisEngine(model, build_dir=BUILD_DIR)
        self.analysis_mode = True

        choice = self.setup_color_var.get()
        self.engine_plays_black_var.set(choice == "engine_black")
        self.engine_plays_white_var.set(choice == "engine_white")

        self._sync_analysis_position()

        self._show_panel(self.analysisResultPanel)
        self._update_status()
        self.root.after(100, self.poll_analysis)

    def _sync_analysis_position(self):
        """Replay the current game's move sequence into the analysis engine, then leave it
        paused -- analysis only actually starts once the user presses Resume."""
        self.selected_variation_move = None
        self._reset_debug_mode()
        self._auto_move_seq_len = -1
        self.analysis_paused = True
        self.analysisPauseButton.configure(text="Resume")
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

    def apply_analysis_limit(self):
        """Re-request analysis with whatever limit is currently in the entry box.

        The engine tracks visits cumulatively on the current position, so raising the
        limit resumes search on the existing tree instead of restarting; if the limit
        is at or below what's already been searched, the engine just stops (or was
        already stopped) and re-prints the current snapshot as-is -- it never resets.
        """
        if not self.analysis_mode or self.analysisEngine is None:
            return
        self.analysis_paused = False
        self.analysisPauseButton.configure(text="Pause")
        self.analysisEngine.request_analysis(self._get_analysis_limit())

    def toggle_analysis_pause(self):
        """Pause halts the engine's in-progress search after its current chunk, keeping
        the tree as-is; Resume just re-sends the current limit, which continues search
        on the existing (unreset) tree from wherever it left off.
        """
        if not self.analysis_mode or self.analysisEngine is None:
            return
        if self.analysis_paused:
            self.analysis_paused = False
            self.analysisPauseButton.configure(text="Pause")
            self.analysisEngine.request_analysis(self._get_analysis_limit())
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
        if not self.analysis_mode or self.game_over:
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
                self._draw_variation_overlay(moves)
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

                    color = _blend_hex(BOARD_BG, OVERLAY_COLOR, val / max_val)
                    text_color = "#ffffff" if _luminance(color) < 140 else "#000000"
                    label = f"{val * 100:.0f}%" if mode == "policy" else f"{int(val)}"

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

    def _draw_variation_overlay(self, moves):
        """KataGo-style principal-variation overlay: numbered stone-colored markers tracing the
        expected continuation for the selected candidate (or the top/highest-visit move if none
        has been clicked in the list yet), alternating color by ply."""
        target = None
        if self.selected_variation_move is not None:
            target = next((mv for mv in moves if mv["move"] == self.selected_variation_move
                            and mv["visits"] > 0), None)
        if target is None:
            target = max(moves, key=lambda mv: mv["visits"])
        if target["visits"] <= 0:
            return

        seq = [target["move"]] + target.get("variation", [])
        self._draw_numbered_path_overlay(seq)

    def _draw_playout_path_overlay(self):
        """Debug mode's Variation-overlay override: the selected playout's actual root->leaf
        search path (from our own accumulated playout_log -- see poll_analysis), drawn the same
        way as the ordinary engine-line variation above -- just from a different, per-playout
        source."""
        if self.selected_playout_index is None or self.selected_playout_index >= len(self.playout_log):
            return
        self._draw_numbered_path_overlay(self.playout_log[self.selected_playout_index]["path"])

    def _draw_numbered_path_overlay(self, seq):
        """Numbered stone-colored markers tracing a move sequence starting from whoever is
        currently to move, alternating color by ply -- the shared drawing routine behind both
        the ordinary engine-line variation and debug mode's per-playout path."""
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

            cx = self.m + x * self.delta
            cy = self.m + y * self.delta
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline="#000000",
                                     tags=("move_overlay", "move_overlay_shape"))
            self.canvas.create_text(cx, cy, text=str(idx + 1), fill=text_color,
                                     font=("Helvetica", font_size, "bold"),
                                     tags=("move_overlay", "move_overlay_text"))

        self.canvas.tag_lower("move_overlay_shape")
        self.canvas.tag_raise("move_overlay_text")

    def _update_analysis_result_panel(self, result):
        self.last_analysis_result = result
        self._draw_move_overlay(result)

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
            limit = self._get_analysis_limit()
            self.analysisVisitsLabel.configure(text=f"Visits: {visits} / {limit} ({status})")

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
        if visits >= self._get_analysis_limit():
            return "done"
        return "analyzing..."

    def _update_candidate_listbox(self, result):
        show_variation = self.overlay_var.get() == "variation"

        if show_variation:
            self.candidateListHeader.configure(text="Candidate moves, with follow-up line:")
            self.candidateColumnHeader.configure(text="expected continuation")
        else:
            self.candidateListHeader.configure(text="Candidate moves, by visit count:")
            self.candidateColumnHeader.configure(text="move       visits   policy  winrate     Q")

        self.analysisListbox.delete(0, END)
        moves = _sorted_candidate_moves(result)
        selected_row = None
        for idx, mv in enumerate(moves):
            r, c = mv["move"]
            if mv["move"] == self.selected_variation_move:
                selected_row = idx
            if show_variation:
                seq = [(r, c)] + mv.get("variation", [])
                self.analysisListbox.insert(END, " → ".join(f"({sr},{sc})" for sr, sc in seq))
            else:
                pct = (mv["winrate"] + 1) / 2 * 100
                self.analysisListbox.insert(
                    END, f"({r},{c})".ljust(10) + f"{int(mv['visits']):<8}"
                         f"{mv['prior'] * 100:5.1f}%  {pct:5.1f}%  {mv['q']:+.3f}")

        if show_variation and selected_row is not None:
            self.analysisListbox.selection_set(selected_row)

    def _on_candidate_listbox_select(self, event=None):
        if self.overlay_var.get() != "variation" or self.last_analysis_result is None:
            return
        sel = self.analysisListbox.curselection()
        if not sel:
            return
        moves = _sorted_candidate_moves(self.last_analysis_result)
        idx = sel[0]
        if idx >= len(moves):
            return
        self.selected_variation_move = moves[idx]["move"]
        self._draw_move_overlay(self.last_analysis_result)

    def _on_debug_mode_change(self):
        """Debug mode only streams playouts run *after* it's turned on -- the engine keeps
        nothing from before, so a stale "done" position would otherwise show an empty playout
        list until the user separately bumped the playout limit. Force a small fresh burst of
        search right away instead, so turning debug mode on always produces something to look
        at. Toggling either way also starts us over locally: our own accumulated playout_log
        described whatever was captured under the *previous* on/off state, which the freshly
        (un)toggled engine process's stream no longer matches.
        """
        self.playout_log = []
        self.selected_playout_index = None
        on = self.debug_mode_var.get()

        self.debugFrame.pack_forget()
        self.candidateFrame.pack_forget()
        (self.debugFrame if on else self.candidateFrame).pack(fill=BOTH, expand=True)

        if self.analysisEngine is None:
            return
        self.analysisEngine.send_debug(on)
        if on:
            current_visits = (self.last_analysis_result or {}).get("visits") or 0
            target = max(self._get_analysis_limit(), current_visits + 200)
            self.analysis_limit_var.set(str(target))
            self.analysis_paused = False
            self.analysisPauseButton.configure(text="Pause")
            self.analysisEngine.request_analysis(target)
        if self.last_analysis_result is not None:
            self._draw_move_overlay(self.last_analysis_result)
        self._update_debug_listbox()

    def _update_debug_listbox(self):
        self.debugListbox.delete(0, END)
        selected_row = None
        for row, rec in enumerate(self.playout_log):
            if row == self.selected_playout_index:
                selected_row = row
            path_text = " ".join(f"({r},{c})" for r, c in rec["path"]) or "(root)"
            if rec["forced"] != 0:
                stat_text = "forced win" if rec["forced"] > 0 else "forced loss"
            else:
                stat_text = f"{rec['winp']:+.2f}  {rec['score']:+.2f}"
            self.debugListbox.insert(END, f"{row:<5} {path_text:<34} {stat_text}")
        if selected_row is not None:
            self.debugListbox.selection_set(selected_row)
            self.debugListbox.see(selected_row)

    def _on_debug_listbox_select(self, event=None):
        sel = self.debugListbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.playout_log):
            return
        self.selected_playout_index = idx
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
        move surfaced by MCTS::printAnalysis, so the same "most visits" pick applies there
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

    def on_end(self, winner):
        self.game_over = True
        if winner == 1:
            self.statusLabel.configure(text="winner is black")
        elif winner == -1:
            self.statusLabel.configure(text="winner is white")
        else:
            result = self.rule.end_game()
            if result[0] == 1:
                self.statusLabel.configure(text=f"winner is black by {result[1]}")
            elif result[0] == -1:
                self.statusLabel.configure(text=f"winner is white by {result[1]}")
            else:
                self.statusLabel.configure(text="draw")

        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None

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
