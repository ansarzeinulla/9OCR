"""Togyzkumalak rules engine, ported from the 9Q C++ engine
(~/Desktop/9Q/src/togyzkumalak_rules.cpp) and validated against its
shortest-game replay fixture (tests/test_rules.py).

Notation matches 9Q and the scoresheet classifier classes:
"{source_hole}{landing_column}" with an "x" suffix when the move creates a
tuzdyk, e.g. "98", "13x". Source hole and landing column are 1-9; the landing
column is the pit's index within its own side, regardless of side.
"""

from dataclasses import dataclass, field

WIN_THRESHOLD = 82
PITS_PER_SIDE = 9


@dataclass
class Move:
    action: int  # 0-8 source hole index on the mover's side
    notation: str  # e.g. "98" or "13x"
    captured: int  # stones captured by the landing rule (excl. tuzdyk income)
    makes_tuzdyk: bool


@dataclass
class Game:
    pits: list[int] = field(default_factory=lambda: [9] * 18)  # P1: 0-8, P2: 9-17
    kazans: list[int] = field(default_factory=lambda: [0, 0])
    # tuzdyks[p] = column (0-8) OWNED BY p on the opponent's side, -1 = none
    tuzdyks: list[int] = field(default_factory=lambda: [-1, -1])
    to_play: int = 0

    def copy(self) -> "Game":
        return Game(self.pits[:], self.kazans[:], self.tuzdyks[:], self.to_play)

    # --- core mechanics -------------------------------------------------

    def _side_of(self, pit: int) -> int:
        return pit // PITS_PER_SIDE

    def _apply(self, action: int) -> Move:
        """Apply a (legal) move for `to_play`; returns the Move played."""
        p = self.to_play
        opp = 1 - p
        start = p * PITS_PER_SIDE + action
        stones = self.pits[start]

        if stones == 1:
            self.pits[start] = 0
            last = (start + 1) % 18
            self.pits[last] += 1
        else:
            self.pits[start] = 1
            pit = start
            for _ in range(stones - 1):
                pit = (pit + 1) % 18
                self.pits[pit] += 1
            last = pit

        # stones sown into a tuzdyk go to its owner's kazan
        for owner in (0, 1):
            col = self.tuzdyks[owner]
            if col >= 0:
                tuz_pit = (1 - owner) * PITS_PER_SIDE + col
                if self.pits[tuz_pit]:
                    self.kazans[owner] += self.pits[tuz_pit]
                    self.pits[tuz_pit] = 0

        captured = 0
        makes_tuzdyk = False
        if self._side_of(last) == opp:
            col = last % PITS_PER_SIDE
            count = self.pits[last]
            if (
                count == 3
                and self.tuzdyks[p] == -1
                and col != PITS_PER_SIDE - 1  # hole 9 can never become a tuzdyk
                and self.tuzdyks[opp] != col  # no mirrored tuzdyks
            ):
                self.tuzdyks[p] = col
                self.kazans[p] += 3
                self.pits[last] = 0
                makes_tuzdyk = True
            elif count % 2 == 0:
                captured = count
                self.kazans[p] += count
                self.pits[last] = 0

        notation = f"{action + 1}{(last % PITS_PER_SIDE) + 1}"
        if makes_tuzdyk:
            notation += "x"
        self.to_play = opp
        return Move(action, notation, captured, makes_tuzdyk)

    # --- public API -----------------------------------------------------

    def legal_actions(self) -> list[int]:
        p = self.to_play
        opp = 1 - p
        return [
            a
            for a in range(PITS_PER_SIDE)
            if self.pits[p * PITS_PER_SIDE + a] > 0 and self.tuzdyks[opp] != a
        ]

    def legal_moves(self) -> list[Move]:
        """The <=9 legal moves with their exact notation (via simulation)."""
        return [self.copy()._apply(a) for a in self.legal_actions()]

    def play(self, action: int) -> Move:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action + 1} for player {self.to_play + 1}")
        return self._apply(action)

    def play_notation(self, notation: str) -> Move:
        """Play by full notation string; raises ValueError if not legal."""
        for move in self.legal_moves():
            if move.notation == notation:
                return self._apply(move.action)
        raise ValueError(f"move {notation!r} is not legal here")

    @property
    def is_over(self) -> bool:
        return (
            max(self.kazans) >= WIN_THRESHOLD
            or not self.legal_actions()
        )

    def winner(self) -> int:
        """0/1 = player, -1 = draw. If the side to move is stuck (atsyrau),
        the opponent collects the stones remaining on their own side."""
        kazans = self.kazans[:]
        if max(kazans) < WIN_THRESHOLD and not self.legal_actions():
            opp = 1 - self.to_play
            kazans[opp] += sum(
                self.pits[opp * PITS_PER_SIDE : (opp + 1) * PITS_PER_SIDE]
            )
        if kazans[0] == kazans[1]:
            return -1
        return 0 if kazans[0] > kazans[1] else 1

    def side_totals(self) -> tuple[int, int]:
        return sum(self.pits[:PITS_PER_SIDE]), sum(self.pits[PITS_PER_SIDE:])
