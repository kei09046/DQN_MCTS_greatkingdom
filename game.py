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

from tkinter import *
from PIL import Image, ImageTk

# 실제 게임이 실행되는 파일 GUI 담당이기도 함

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

# engine Color enum (src/consts.h): BLACK = 1, WHITE = 2
ENGINE_BLACK = 1
ENGINE_WHITE = 2

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
        self.root.geometry("1300x1000")

        self.boardFrame = Frame(self.root)
        self.canvas = Canvas(self.boardFrame, width=900, height=900, bg="#DCB35C", highlightthickness=0)
        self.passButton = Button(self.boardFrame, text="pass")
        self.statusLabel = Label(self.boardFrame, text="", font=("Helvetica", 14))

        self.sideFrame = Frame(self.root, width=300)
        Label(self.sideFrame, text="Engine Analysis", font=("Helvetica", 16, "bold")) \
            .pack(anchor="w", pady=(0, 15))
        self.variationLabel = Label(self.sideFrame, text="Top line: -", font=("Helvetica", 11),
                                     wraplength=260, justify=LEFT, anchor="w")
        self.variationLabel.pack(fill=X, pady=5, anchor="w")
        self.winProbLabel = Label(self.sideFrame, text="Win probability: -", font=("Helvetica", 12), anchor="w")
        self.winProbLabel.pack(fill=X, pady=5, anchor="w")
        self.scoreLabel = Label(self.sideFrame, text="Expected score diff: -", font=("Helvetica", 12), anchor="w")
        self.scoreLabel.pack(fill=X, pady=5, anchor="w")
        self.moveTimeLabel = Label(self.sideFrame, text="Move time: -", font=("Helvetica", 12), anchor="w")
        self.moveTimeLabel.pack(fill=X, pady=5, anchor="w")

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
        self.setupFrame = None

    def show_setup_screen(self):
        models = list_models()

        frame = Frame(self.root)
        frame.pack(expand=True)
        self.setupFrame = frame

        Label(frame, text="Great Kingdom", font=("Helvetica", 20)).pack(pady=(30, 15))

        Label(frame, text="Model:").pack()
        model_var = StringVar(value=models[0] if models else "")
        if models:
            OptionMenu(frame, model_var, *models).pack(pady=(0, 15))
        else:
            Label(frame, text=f"(no .pt files found in {MODELS_DIR})", fg="red").pack(pady=(0, 15))

        Label(frame, text="Play as:").pack()
        color_var = StringVar(value="black")
        Radiobutton(frame, text="Black", variable=color_var, value="black").pack()
        Radiobutton(frame, text="White", variable=color_var, value="white").pack()

        def start():
            model = model_var.get()
            if not model:
                return
            frame.destroy()
            self.setupFrame = None
            self.play_ai(human_color=(0 if color_var.get() == "black" else 1), model=model)

        start_button = Button(frame, text="Start Game", command=start)
        start_button.pack(pady=25)
        if not models:
            start_button.configure(state=DISABLED)

    def create_board(self, ag_ai=False):
        self.boardFrame.grid(row=0, column=0, sticky="n")
        self.canvas.pack()
        self.passButton.pack()
        self.passButton.configure(command=self.on_pass)
        self.statusLabel.pack()

        for i in range(self.boardSize):
            self.canvas.create_line(i * self.delta + self.m, self.m, i * self.delta + self.m,
                               (self.boardSize - 1) * self.delta + self.m)
            self.canvas.create_line(self.m, i * self.delta + self.m, (self.boardSize - 1) * self.delta + self.m,
                               i * self.delta + self.m)

        self._draw_star_points()

        for i in self.neutral:
            self.canvas.create_image(self.m + i[0] * self.delta - self.stone_size / 2,
                                     self.m + i[1] * self.delta - self.stone_size / 2,
                        anchor=NW, image=self.stones[2])
            self.on_stone[i[0]][i[1]] = True

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                self.cord[i][j] = (self.m + i * self.delta, self.m + j * self.delta)

        self.against_ai = ag_ai
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if self.against_ai:
            self.sideFrame.grid(row=0, column=1, sticky="n", padx=20, pady=20)
            self._update_status()
            self.root.after(100, self.poll_engine)
            self.root.after(100, self.poll_stats)

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
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="black", outline="black")

    def _place_stone(self, x, y, color_idx):
        x_loc = self.m + x * self.delta
        y_loc = self.m + y * self.delta
        self.canvas.create_image(x_loc - self.stone_size / 2, y_loc - self.stone_size / 2,
                                 anchor=NW, image=self.stones[color_idx])
        self.on_stone[x][y] = True

    def _update_status(self):
        if self.game_over:
            return
        if not self.against_ai:
            self.statusLabel.configure(text="")
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

            if result == -2:
                self.on_end(-2)
                return

            self.turn = 1 - self.turn
            self._update_status()
            return

    # human_color: 0 for black, 1 for white
    def play_ai(self, human_color=0, model='H96000B.pt', build_dir=BUILD_DIR):
        self.human_color = human_color
        engine_color = ENGINE_BLACK if human_color == 0 else ENGINE_WHITE
        self.engine = EngineProcess(model, engine_color, build_dir=build_dir)
        self.create_board(ag_ai=True)

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

    def on_close(self):
        if self.engine is not None:
            self.engine.close()
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
    g.show_setup_screen()
    g.root.mainloop()
