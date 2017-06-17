
# --- GO MAIN PROGRAM DEPENDENCIES ---
from __future__ import print_function
from collections import namedtuple
from itertools import count
import re
import random
import sys
import copy

# === EXECUTABLE AND TIMEOUT DEPENDENCIES ---
import time
from multiprocessing.pool import ThreadPool
from multiprocessing import TimeoutError
import subprocess as sp
import os


# variables
N = 9
W = N + 2
empty = "\n".join([(N+1)*' '] + N*[' '+N*'.'] + [(N+2)*' '])
colstr = 'ABCDEFGHIJKLMNOPQRST'


# board string routines
def neighbors(c):
    """ generator of coordinates for all neighbors of c """
    return [c-1, c+1, c-W, c+W]
def diag_neighbors(c):
    """ generator of coordinates for all diagonal neighbors of c """
    return [c-W-1, c-W+1, c+W-1, c+W+1]
def board_put(board, c, p):
    return board[:c] + p + board[c+1:]
def floodfill(board, c):
    """ replace continuous-color area starting at c with special color # """
    # This is called so much that a bytearray is worthwhile...
    byteboard = bytearray(board)
    p = byteboard[c]
    byteboard[c] = ord('#')
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if byteboard[d] == p:
                byteboard[d] = ord('#')
                fringe.append(d)
    return str(byteboard)
# Regex that matches various kind of points adjecent to '#' (floodfilled) points
contact_res = dict()
for p in ['.', 'x', 'X']:
    rp = '\\.' if p == '.' else p
    contact_res_src = ['#' + rp,  # p at right
                       rp + '#',  # p at left
                       '#' + '.'*(W-1) + rp,  # p below
                       rp + '.'*(W-1) + '#']  # p above
    contact_res[p] = re.compile('|'.join(contact_res_src), flags=re.DOTALL)
def contact(board, p):
    """ test if point of color p is adjecent to color # anywhere
    on the board; use in conjunction with floodfill for reachability """
    m = contact_res[p].search(board)
    if not m:
        return None
    return m.start() if m.group(0)[0] == p else m.end() - 1
def is_eyeish(board, c):
    """ test if c is inside a single-color diamond and return the diamond
    color or None; this could be an eye, but also a false one """
    eyecolor = None
    for d in neighbors(c):
        if board[d].isspace():
            continue
        if board[d] == '.':
            return None
        if eyecolor is None:
            eyecolor = board[d]
            othercolor = eyecolor.swapcase()
        elif board[d] == othercolor:
            return None
    return eyecolor
def is_eye(board, c):
    """ test if c is an eye and return its color or None """
    eyecolor = is_eyeish(board, c)
    if eyecolor is None:
        return None

    # Eye-like shape, but it could be a falsified eye
    falsecolor = eyecolor.swapcase()
    false_count = 0
    at_edge = False
    for d in diag_neighbors(c):
        if board[d].isspace():
            at_edge = True
        elif board[d] == falsecolor:
            false_count += 1
    if at_edge:
        false_count += 1
    if false_count >= 2:
        return None

    return eyecolor
class Position(namedtuple('Position', 'board cap n ko last last2 komi')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    def move(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def checkmove(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return False
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return False

        # Update the position and return
        return True


    def pass_move(self):
        """ pass - i.e. return simply a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                        n=self.n + 1, ko=None, last=None, last2=self.last, komi=self.komi)

    def moves(self, i0):
        """ Generate a list of moves (includes false positives - suicide moves;
        does not include true-eye-filling moves), starting from a given board
        index (that can be used for randomization) """
        i = i0-1
        passes = 0
        while True:
            i = self.board.find('.', i+1)
            if passes > 0 and (i == -1 or i >= i0):
                break  # we have looked through the whole board
            elif i == -1:
                i = 0
                passes += 1
                continue  # go back and start from the beginning
            # Test for to-play player's one-point eye
            if is_eye(self.board, i) == 'X':
                continue
            yield i

    def last_moves_neighbors(self):
        """ generate a randomly shuffled list of points including and
        surrounding the last two moves (but with the last move having
        priority) """
        clist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist = [c] + list(neighbors(c) + diag_neighbors(c))
            random.shuffle(dlist)
            clist += [d for d in dlist if d not in clist]
        return clist

    def score(self, owner_map=None):
        """ compute score for to-play player; this assumes a final position
        with all dead stones captured; if owner_map is passed, it is assumed
        to be an array of statistics with average owner at the end of the game
        (+1 black, -1 white) """
        board = self.board
        i = 0
        while True:
            i = self.board.find('.', i+1)
            if i == -1:
                break
            fboard = floodfill(board, i)
            # fboard is board with some continuous area of empty space replaced by #
            touches_X = contact(fboard, 'X') is not None
            touches_x = contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
            # now that area is replaced either by X, x or :
        komi = self.komi if self.n % 2 == 1 else -self.komi
        if owner_map is not None:
            for c in range(W*W):
                n = 1 if board[c] == 'X' else -1 if board[c] == 'x' else 0
                owner_map[c] += n * (1 if self.n % 2 == 0 else -1)
        return board.count('X') - board.count('x') + komi

    def area(self, owner_map=None):
        board = self.board
        i = 0
        while True:
            i = self.board.find('.', i+1)
            if i == -1:
                break
            fboard = floodfill(board, i)
            # fboard is board with some continuous area of empty space replaced by #
            touches_X = contact(fboard, 'X') is not None
            touches_x = contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
            # now that area is replaced either by X, x or :
        komi = self.komi if self.n % 2 == 1 else -self.komi
        if owner_map is not None:
            for c in range(W*W):
                n = 1 if board[c] == 'X' else -1 if board[c] == 'x' else 0
                owner_map[c] += n * (1 if self.n % 2 == 0 else -1)
        return [board.count('X'),board.count('x')]

def empty_position():
    """ Return an initial board position """
    return Position(board=empty, cap=(0, 0), n=0, ko=None, last=None, last2=None, komi=7.5)

# Utility routines
def parse_board(tmp):
    arr = tmp.split("\n")
    board = []
    start = 1
    for i in range(0, N):
        tmp = []
        for j in range(0, N):
            tmp.append(arr[start + i][j + 1])
        board.append(tmp)
    return board
def print_board(arr):
    str = ""
    n = len(arr)
    for i in range(0, n):
        for j in range(0, n):
            str += arr[i][j]
        str += "\n"
    print(str)
def print_pos(pos, f=sys.stderr, owner_map=None):
    """ print visualization of the given board position, optionally also
    including an owner map statistic (probability of that area of board
    eventually becoming black/white) """
    if pos.n % 2 == 0:  # to-play is black
        board = pos.board.replace('x', 'O')
        Xcap, Ocap = pos.cap
    else:  # to-play is white
        board = pos.board.replace('X', 'O').replace('x', 'X')
        Ocap, Xcap = pos.cap
    print('Move: %-3d   Black: %d caps   White: %d caps  Komi: %.1f' % (pos.n, Xcap, Ocap, pos.komi), file=f)
    pretty_board = ' '.join(board.rstrip()) + ' '
    if pos.last is not None:
        pretty_board = pretty_board[:pos.last*2-1] + '(' + board[pos.last] + ')' + pretty_board[pos.last*2+2:]
    rowcounter = count()
    pretty_board = [' %-02d%s' % (N-i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)]

    print("\n".join(pretty_board), file=f)
    print('    ' + ' '.join(colstr[:N]), file=f)
    print('', file=f)
def parse_coord(s):
    if s == 'pass':
        return None
    try:
        res = W+1 + (N - int(s[1:])) * W + colstr.index(s[0].upper())
    except ValueError:
        res = "wi"
    return res
def str_coord(c):
    if c is None:
        return 'pass'
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], N - row)



# Game Properties
class Simulation:
    def __init__(self, pos):
        self.pos = pos
def execute(cmd, prog, sim):
    #-- input test (manual input to check rule violation) --
    #sc = raw_input()

    #--- program ---
    proc = sp.Popen([cmd, prog, sim.pos.board], stdout=sp.PIPE)
    sc = proc.stdout.read()

    cont = True
    while(cont):
        cont = False
        args = None

        print("player sends: "+ sc)
        c = parse_coord(sc)
        if c is "wi":
            # wrong input
            cont = True
            args = "err0"
        elif sim.pos.board[c] != '.':
            # position not empty
            cont = True
            args = "err1"
        elif not sim.pos.move(c):
            # rule violation
            cont = True
            args = "err2"
        if(cont):
            print("checker returns " + args)

            # -- program --
            proc = sp.Popen([cmd, prog, args], stdout=sp.PIPE)
            sc = proc.stdout.read()

            # -- manual input to check rule violation --
            #sc = raw_input()

    return c
def check(cmd, prog, sim, timeout):
    start = time.time()

    pool = ThreadPool(processes=1)
    a = pool.apply_async(execute, (cmd,prog,sim,))
    c = None
    try:
        c = a.get(timeout=timeout)
    except TimeoutError:
        print("timeout")

    end = time.time()
    print("Elapsed: " + str(end - start))

    return c

def isDone(sim):
    for i in range (0, len(sim.pos.board)):
        if isValid(i, sim):
            return False
    return True

def isValid(c, sim):
    if sim.pos.board[c] != '.':
        # position not empty
        return False
    elif not sim.pos.move(c):
        # rule violation
        return False
    return True

def game_io(cmd, prog, timeout):
    """ A simple minimalistic text mode UI. """

    sim = Simulation(pos=empty_position())
    owner_map = W*W*[0]
    res = []
    score = [0,0]
    passes = [0,0]
    win = None

    while True:
        print_pos(sim.pos, sys.stdout, owner_map)
        # print(sim.pos.board)
        # board = parse_board(sim.pos.board)
        # print_board(board)
        index = sim.pos.n % 2
        print("player", index + 1)

        # check timeout
        c = check(cmd[index], prog[index], sim, timeout)

        if c is not None:
            # Not a pass
            sim = Simulation(pos=sim.pos.move(c))

            # --- check if next player cannot move (game is done)---
            if isDone(sim):
                print_pos(sim.pos, sys.stdout, owner_map)
                res = sim.pos.area()
                print(res)
                if(res[1] > res[0]):
                    win = index
                elif(res[0] > res[1]):
                    win = 1-index
                else:
                    win = 2
                break
        else:
            # Pass move
            print("\n PASS")

            passes[index] += 1
            sim = Simulation(pos=sim.pos.pass_move())

            # --- if any passes 3 times ---
            print(passes)
            if passes[index] >= 3:
                if index % 2 == 0:
                    win = 1
                else:
                    win = 0
                break

        # # --- calculate area won ---
        # res = sim.pos.Area()
        # if index == 0:
        #     score[0] = res[0]
        #     score[1] = res[1]
        # else:
        #     score[1] = res[0]
        #     score[0] = res[1]
        # print("player 1 score = ", score[0], "| player 2 score = ", score[1])



    if win == 0:
        print("player 1 wins")
    elif win == 1:
        print("player 2 wins")
    else:
        print("draw")


    print("Thanks For Playing")



cmd = ["python", "python"]
cwd = os.getcwd()

temp2 = cwd+"\\test.py"
temp1 = cwd+"\\test.py"
prog = [temp1, temp2]





game_io(cmd, prog, 180)