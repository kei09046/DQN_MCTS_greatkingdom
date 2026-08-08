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

# engine Color enum (src/consts.h): BLACK = 1, WHITE = 2
ENGINE_BLACK = 1
ENGINE_WHITE = 2

BOARD_BG = "#DCB35C"
OVERLAY_COLOR = "#1565C0"  # blended toward this color for stronger policy/visit intensity


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

# "move : <r> <c>" -- the engine's chosen move
_MOVE_RE = re.compile(r"^move\s*:\s*(-?\d+)\s+(-?\d+)")
# "move time : <microseconds>[µs]"
_MOVETIME_RE = re.compile(r"^move time\s*:\s*(\d+)")
# one line of MCTS::printVariation(): "<r> <c> <visit> forced <fs> Q: ..."
_VARIATION_RE = re.compile(r"^(-?\d+) (-?\d+) (\d+) forced (-?\d+) Q:")
# "winprob : <float>" (-1 = 0%, 1 = 100%, from the engine's perspective)
_WINPROB_RE = re.compile(r"^winprob\s*:\s*" + _FLOAT)
# "scoreEXP : <float>" expected score difference
_SCOREEXP_RE = re.compile(r"^scoreEXP\s*:\s*" + _FLOAT)

# ---- ModelCompare::analyze() protocol (see MCTS::printAnalysis in src/PMCTS.cpp) ----
# "winrate : <float>" (-1 = 0%, 1 = 100%, from the perspective of whoever is to move)
_ANALYSIS_WINRATE_RE = re.compile(r"^winrate\s*:\s*" + _FLOAT)
# "visits : <int>" total root visit count
_ANALYSIS_VISITS_RE = re.compile(r"^visits\s*:\s*(\d+)")
# "initQ : <float>" -- the raw value-head output the first time this position was evaluated
_ANALYSIS_INITQ_RE = re.compile(r"^initQ\s*:\s*" + _FLOAT)
# "move <r> <c> visits <N> prior <P> winrate <W> q <Q> variation <r1> <c1> <r2> <c2> ..." --
# one candidate move; the variation tail is 0 or more coordinate pairs, captured whole and
# split separately since its length varies per move.
_ANALYSIS_MOVE_RE = re.compile(
    r"^move\s+(-?\d+)\s+(-?\d+)\s+visits\s+" + _FLOAT + r"\s+prior\s+" + _FLOAT
    + r"\s+winrate\s+" + _FLOAT + r"\s+q\s+" + _FLOAT + r"\s+variation(.*)$")


def list_models(models_dir=MODELS_DIR):
    if not os.path.isdir(models_dir):
        return []
    return sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))


class EngineProcess:
    """Wraps `./play play <model> <humanColor>` and talks to it over stdin/stdout.

    The engine blocks on stdin only while it's the human's turn; once it becomes
    the engine's turn it computes and prints "move : r c" on its own, so moves
    coming back from the engine are picked up asynchronously via a reader thread.
    """

    def __init__(self, model, human_color, build_dir=BUILD_DIR):
        self.proc = subprocess.Popen(
            ["./play", "play", model, str(human_color)],
            cwd=build_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # moves: (x, y, move_time_us) tuples for engine-chosen moves
        self.moves = queue.Queue()
        # stats: dicts {"variation": [(r,c), ...], "winprob": float, "score": float},
        # pushed progressively as the engine deepens its search for the current move
        self.stats = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        variation_buf = []
        pending_winprob = None
        pending_move = None

        for line in self.proc.stdout:
            line = line.rstrip("\n")

            m = _VARIATION_RE.match(line)
            if m:
                variation_buf.append((int(m.group(1)), int(m.group(2))))
                continue

            m = _WINPROB_RE.match(line)
            if m:
                pending_winprob = float(m.group(1))
                continue

            m = _SCOREEXP_RE.match(line)
            if m:
                self.stats.put({
                    "variation": variation_buf,
                    "winprob": pending_winprob,
                    "score": float(m.group(1)),
                })
                variation_buf = []
                pending_winprob = None
                continue

            m = _MOVETIME_RE.match(line)
            if m and pending_move is not None:
                self.moves.put(pending_move + (int(m.group(1)),))
                pending_move = None
                continue

            m = _MOVE_RE.match(line)
            if m:
                pending_move = (int(m.group(1)), int(m.group(2)))
                continue

    def send_move(self, x, y):
        if self.proc.poll() is not None or self.proc.stdin.closed:
            return
        try:
            self.proc.stdin.write(f"{x} {y}\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def get_move_nowait(self):
        try:
            return self.moves.get_nowait()
        except queue.Empty:
            return None

    def get_stats_nowait(self):
        try:
            return self.stats.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


class AnalysisEngine:
    """Wraps `./play analyze <model>` (ModelCompare::analyze in src/modelcompare.cpp).

    Unlike EngineProcess, this never moves on its own: the caller drives the position
    with reset()/play() and explicitly asks for a winrate/policy/visit-count breakdown
    of every candidate move via request_analysis(). Results stream back asynchronously
    through a reader thread, same pattern as EngineProcess.
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
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        in_block = False
        current = None

        for line in self.proc.stdout:
            line = line.rstrip("\n")

            if line == "analysis begin":
                in_block = True
                current = {"winrate": None, "visits": None, "initQ": None, "moves": []}
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

    def get_result_nowait(self):
        try:
            return self.results.get_nowait()
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
        self.analysisPanel = None
        self.analysisResultPanel = None
        self._build_setup_panel()
        self._build_analysis_panel()
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
        self.against_ai = False
        self.engine = None
        self.human_color = 0
        self.game_over = False
        self.analysis_mode = False
        self.analysisEngine = None
        self.last_analysis_result = None
        self.selected_variation_move = None

    def _build_setup_panel(self):
        panel = Frame(self.sideFrame)
        self.setupPanel = panel

        Label(panel, text="Great Kingdom", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 15))

        Label(panel, text="Play vs an engine", font=("Helvetica", 12, "bold"), anchor="w").pack(fill=X)

        models = list_models()
        Label(panel, text="Model:", anchor="w").pack(fill=X, pady=(10, 0))
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

        Label(panel, text="Play as:", anchor="w").pack(fill=X)
        self.setup_color_var = StringVar(value="black")
        Radiobutton(panel, text="Black", variable=self.setup_color_var, value="black", anchor="w") \
            .pack(fill=X)
        Radiobutton(panel, text="White", variable=self.setup_color_var, value="white", anchor="w") \
            .pack(fill=X)

        self.start_ai_button = Button(panel, text="Start Engine Game", command=self.start_ai_game)
        self.start_ai_button.pack(fill=X, pady=(15, 0))
        if not models:
            self.start_ai_button.configure(state=DISABLED)

        Frame(panel, height=1, bg="#999999").pack(fill=X, pady=20)

        Label(panel, text="No engine selected: the board is\nfree to play locally, move by move.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(0, 10))
        Button(panel, text="New Local Game", command=self.reset_board).pack(fill=X)

        Frame(panel, height=1, bg="#999999").pack(fill=X, pady=20)

        Label(panel, text="See what the selected model thinks of\nthe position currently on the board.",
              wraplength=260, justify=LEFT, fg="#555555", anchor="w").pack(fill=X, pady=(0, 10))
        self.analyze_button = Button(panel, text="Analyze Position", command=self.start_position_analysis)
        self.analyze_button.pack(fill=X)
        if not models:
            self.analyze_button.configure(state=DISABLED)

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

    def _build_analysis_panel(self):
        panel = Frame(self.sideFrame)
        self.analysisPanel = panel

        Label(panel, text="Engine Analysis", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 15))
        self.variationLabel = Label(panel, text="Top line: -", font=("Helvetica", 11),
                                     wraplength=260, justify=LEFT, anchor="w")
        self.variationLabel.pack(fill=X, pady=5, anchor="w")
        self.winProbLabel = Label(panel, text="Win probability: -", font=("Helvetica", 12), anchor="w")
        self.winProbLabel.pack(fill=X, pady=5, anchor="w")
        self.scoreLabel = Label(panel, text="Expected score diff: -", font=("Helvetica", 12), anchor="w")
        self.scoreLabel.pack(fill=X, pady=5, anchor="w")
        self.moveTimeLabel = Label(panel, text="Move time: -", font=("Helvetica", 12), anchor="w")
        self.moveTimeLabel.pack(fill=X, pady=5, anchor="w")

        Button(panel, text="End Game / Back to Setup", command=self.end_ai_game).pack(fill=X, pady=(20, 0))

    def _build_analysis_result_panel(self):
        panel = Frame(self.sideFrame)
        self.analysisResultPanel = panel

        Label(panel, text="Position Analysis", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 15))

        limit_row = Frame(panel)
        limit_row.pack(fill=X, pady=(0, 10))
        Label(limit_row, text="Playout limit:").pack(side=LEFT)
        self.analysis_limit_var = StringVar(value="4000")
        Entry(limit_row, textvariable=self.analysis_limit_var, width=8).pack(side=LEFT, padx=(5, 5))
        Button(limit_row, text="Apply", command=self.apply_analysis_limit).pack(side=LEFT)

        Label(panel, text="Board overlay:", anchor="w").pack(fill=X)
        overlay_row = Frame(panel)
        overlay_row.pack(fill=X, pady=(2, 10))
        self.overlay_var = StringVar(value="none")
        for label, value in (("No effect", "none"), ("Policy", "policy"),
                              ("Visits", "visits"), ("Variation", "variation")):
            Radiobutton(overlay_row, text=label, variable=self.overlay_var, value=value,
                        command=self._on_overlay_mode_change).pack(side=LEFT)

        self.analysisWinrateLabel = Label(panel, text="Win probability: -", font=("Helvetica", 12), anchor="w")
        self.analysisWinrateLabel.pack(fill=X, pady=5, anchor="w")
        self.analysisInitQLabel = Label(panel, text="Initial value (initQ): -", font=("Helvetica", 12), anchor="w")
        self.analysisInitQLabel.pack(fill=X, pady=5, anchor="w")
        self.analysisValueMSELabel = Label(panel, text="Value MSE loss: -", font=("Helvetica", 12), anchor="w")
        self.analysisValueMSELabel.pack(fill=X, pady=5, anchor="w")
        self.analysisPolicyCELabel = Label(panel, text="Policy CE loss: -", font=("Helvetica", 12), anchor="w")
        self.analysisPolicyCELabel.pack(fill=X, pady=5, anchor="w")
        self.analysisVisitsLabel = Label(panel, text="Visits: -", font=("Helvetica", 12), anchor="w")
        self.analysisVisitsLabel.pack(fill=X, pady=5, anchor="w")

        self.candidateListHeader = Label(panel, text="Candidate moves, by visit count:",
                                          font=("Helvetica", 10, "bold"), anchor="w",
                                          wraplength=260, justify=LEFT)
        self.candidateListHeader.pack(fill=X, pady=(10, 0))
        self.candidateColumnHeader = Label(panel, text="move       visits   policy  winrate     Q",
                                            font=("Courier", 9), fg="#555555", anchor="w")
        self.candidateColumnHeader.pack(fill=X, pady=(2, 2))

        list_frame = Frame(panel)
        list_frame.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        self.analysisListbox = Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        font=("Courier", 10), height=20, exportselection=False)
        scrollbar.configure(command=self.analysisListbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.analysisListbox.pack(side=LEFT, fill=BOTH, expand=True)
        self.analysisListbox.bind("<<ListboxSelect>>", self._on_candidate_listbox_select)

        Button(panel, text="Stop Analysis / Back to Setup", command=self.end_analysis_mode).pack(fill=X, pady=(10, 0))

    def _show_panel(self, panel):
        for p in (self.setupPanel, self.analysisPanel, self.analysisResultPanel):
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
        if not self.against_ai:
            self.statusLabel.configure(text="Black to move" if self.turn == 0 else "White to move")
        elif self.human_color == self.turn:
            self.statusLabel.configure(text="your turn")
        else:
            self.statusLabel.configure(text="engine is thinking ...")

    def on_click(self, event):
        if self.game_over:
            return
        if not self.against_ai or (self.human_color == self.turn):
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

            self._place_stone(x_cord, y_cord, self.turn)
            self.turn = 1 - self.turn

            res = self.rule.make_move(x_cord, y_cord)
            if self.against_ai:
                self.engine.send_move(x_cord, y_cord)
            elif self.analysis_mode:
                self.analysisEngine.send_play(x_cord, y_cord)
                self.analysisEngine.request_analysis(self._get_analysis_limit())
                self.selected_variation_move = None
            if res == 1 or res == -1 or res == -2:
                self.on_end(res * self.rule.turn)
            else:
                self._update_status()
        return

    def on_pass(self):
        if self.game_over:
            return
        if not self.against_ai or (self.human_color == self.turn):
            result = self.rule.make_move(-1, 0)
            if self.against_ai:
                self.engine.send_move(self.boardSize, 0)
            elif self.analysis_mode:
                self.analysisEngine.send_play(self.boardSize, 0)
                self.analysisEngine.request_analysis(self._get_analysis_limit())
                self.selected_variation_move = None

            if result == -2:
                self.on_end(-2)
                return

            self.turn = 1 - self.turn
            self._update_status()
            return

    def reset_board(self):
        """Clear the board and start a fresh local (no engine) game."""
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None
        self.canvas.delete("stone")
        self.canvas.delete("move_overlay")
        self.last_analysis_result = None
        self.selected_variation_move = None
        self.rule = RuleManager()
        self.on_stone = [[False for _ in range(self.boardSize)] for _ in range(self.boardSize)]
        self.turn = 0
        self.game_over = False
        self.against_ai = False
        self.analysis_mode = False
        self.human_color = 0
        self.statusLabel.configure(text="")
        self._show_panel(self.setupPanel)
        self._update_status()

    # human_color: 0 for black, 1 for white
    def start_ai_game(self, human_color=None, model=None, build_dir=BUILD_DIR):
        if model is None:
            model = self.setup_model_var.get()
        if not model:
            return
        if human_color is None:
            human_color = 0 if self.setup_color_var.get() == "black" else 1

        self.reset_board()
        self.human_color = human_color
        engine_color = ENGINE_BLACK if human_color == 0 else ENGINE_WHITE
        self.engine = EngineProcess(model, engine_color, build_dir=build_dir)
        self.against_ai = True

        self._show_panel(self.analysisPanel)
        self._update_status()
        self.root.after(100, self.poll_engine)
        self.root.after(100, self.poll_stats)

    def end_ai_game(self):
        self.reset_board()

    def start_position_analysis(self, model=None):
        """Launch (or restart) the analysis engine and evaluate the position currently on the board."""
        if model is None:
            model = self.setup_model_var.get()
        if not model:
            return

        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.against_ai = False
        if self.analysisEngine is not None:
            self.analysisEngine.close()

        self.analysisEngine = AnalysisEngine(model, build_dir=BUILD_DIR)
        self.analysis_mode = True
        self._sync_analysis_position()

        self._show_panel(self.analysisResultPanel)
        self._update_status()
        self.root.after(100, self.poll_analysis)

    def _sync_analysis_position(self):
        """Replay the current game's move sequence into the analysis engine and request a first pass."""
        self.selected_variation_move = None
        self.analysisEngine.send_reset()
        for x, y in self.rule.seq:
            if x == -1 or y == -1:
                self.analysisEngine.send_play(self.boardSize, 0)
            else:
                self.analysisEngine.send_play(x, y)
        self.analysisEngine.request_analysis(self._get_analysis_limit())

    def _get_analysis_limit(self):
        try:
            return max(1, int(self.analysis_limit_var.get()))
        except ValueError:
            return 4000

    def apply_analysis_limit(self):
        """Re-request analysis with whatever limit is currently in the entry box.

        The engine tracks visits cumulatively on the current position, so raising the
        limit resumes search on the existing tree instead of restarting; lowering it
        (or leaving it unchanged) is a cheap no-op re-print of the current snapshot.
        """
        if not self.analysis_mode or self.analysisEngine is None:
            return
        self.analysisEngine.request_analysis(self._get_analysis_limit())

    def end_analysis_mode(self):
        self.analysis_mode = False
        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None
        self.canvas.delete("move_overlay")
        self.last_analysis_result = None
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
        if latest is not None:
            self._update_analysis_result_panel(latest)
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

        mode = self.overlay_var.get()
        if mode == "none":
            return

        moves = result.get("moves", [])
        if not moves:
            return

        if mode == "variation":
            self._draw_variation_overlay(moves)
            return

        if mode == "policy":
            # prior is already a probability (fraction of 1); use it directly as the "share".
            get_val = lambda mv: mv["prior"]
            get_share = lambda mv: mv["prior"]
        else:
            total_visits = result.get("visits") or 0
            if total_visits <= 0:
                return
            get_val = lambda mv: mv["visits"]
            get_share = lambda mv: mv["visits"] / total_visits

        max_val = max((get_val(mv) for mv in moves), default=0.0)
        if max_val <= 0:
            return

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

        # stacking order (bottom to top): overlay circles, grid/star (board coordinates),
        # stones, overlay numbers. Sink the circles below everything already on the board
        # (grid/star/stones) instead of raising grid/star, which would otherwise climb above
        # stones too; keep the numbers on top so they stay legible even over a star point.
        self.canvas.tag_lower("move_overlay_shape")
        self.canvas.tag_raise("move_overlay_text")

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

        mse = _value_mse(result)
        if mse is not None:
            self.analysisValueMSELabel.configure(text=f"Value MSE loss: {mse:.4f}")

        ce = _policy_visit_cross_entropy(result)
        if ce is not None:
            self.analysisPolicyCELabel.configure(text=f"Policy CE loss: {ce:.4f}")

        visits = result.get("visits")
        if visits is not None:
            limit = self._get_analysis_limit()
            status = "analyzing..." if visits < limit else "done"
            self.analysisVisitsLabel.configure(text=f"Visits: {visits} / {limit} ({status})")

        self._update_candidate_listbox(result)

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

    def poll_engine(self):
        if not self.against_ai or self.game_over:
            return
        move = self.engine.get_move_nowait()
        if move is not None:
            self.apply_engine_move(move)
        if not self.game_over:
            self.root.after(100, self.poll_engine)

    def poll_stats(self):
        if not self.against_ai or self.game_over:
            return
        latest = None
        while True:
            s = self.engine.get_stats_nowait()
            if s is None:
                break
            latest = s
        if latest is not None:
            self._update_stats_panel(latest)
        if not self.game_over:
            self.root.after(100, self.poll_stats)

    def _update_stats_panel(self, stats, move_time_us=None):
        variation = stats.get("variation") or []
        if variation:
            var_text = " → ".join(f"({r},{c})" for r, c in variation[:6])
        else:
            var_text = "-"
        self.variationLabel.configure(text=f"Top line: {var_text}")

        winprob = stats.get("winprob")
        if winprob is not None:
            pct = (winprob + 1) / 2 * 100
            self.winProbLabel.configure(text=f"Win probability: {pct:.1f}%")

        score = stats.get("score")
        if score is not None:
            self.scoreLabel.configure(text=f"Expected score diff: {score:+.2f}")

        if move_time_us is not None:
            self.moveTimeLabel.configure(text=f"Move time: {move_time_us / 1e6:.2f}s")

    def apply_engine_move(self, move):
        x, y, move_time_us = move
        if x == self.boardSize and y == 0:  # PASSMOVE = (boardSize, 0)
            result = self.rule.make_move(-1, 0)
            self.turn = 1 - self.turn
            self.moveTimeLabel.configure(text=f"Move time: {move_time_us / 1e6:.2f}s")
            if result == -2:
                self.on_end(-2)
            else:
                self._update_status()
            return

        self._place_stone(x, y, self.turn)
        self.turn = 1 - self.turn
        self.moveTimeLabel.configure(text=f"Move time: {move_time_us / 1e6:.2f}s")

        res = self.rule.make_move(x, y)
        if res == 1 or res == -1 or res == -2:
            self.on_end(res * self.rule.turn)
        else:
            self._update_status()

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

        if self.engine is not None:
            self.engine.close()
        if self.analysisEngine is not None:
            self.analysisEngine.close()
            self.analysisEngine = None

    def on_close(self):
        if self.engine is not None:
            self.engine.close()
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
