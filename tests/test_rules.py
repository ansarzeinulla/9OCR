"""Validate togyz/rules.py against the 9Q engine's shortest-game fixture.

Replays tests/fixtures/shortest_candidate_replay.tsv move by move and asserts
that notation, kazans, side totals, tuzdyks, and the winner all match the
C++ engine's recorded state after every halfmove.

    python tests/test_rules.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from togyz.rules import Game

FIXTURE = Path(__file__).parent / "fixtures" / "shortest_candidate_replay.tsv"


def main() -> None:
    with open(FIXTURE, newline="") as f:
        lines = f.read().splitlines()
    rows = list(csv.DictReader(lines[2:], delimiter="\t"))  # skip 2 meta lines

    game = Game()
    for row in rows:
        expected = row["move"]
        player = 0 if row["player"] == "P1" else 1
        assert game.to_play == player, f"halfmove {row['halfmove']}: wrong side to move"

        legal = {m.notation for m in game.legal_moves()}
        assert expected in legal, f"halfmove {row['halfmove']}: {expected} not in {legal}"

        played = game.play_notation(expected)
        assert played.notation == expected

        state = (game.kazans[0], game.kazans[1], *game.side_totals(),
                 game.tuzdyks[0], game.tuzdyks[1])
        want = tuple(int(row[k]) for k in
                     ("p1_kazan", "p2_kazan", "p1_side", "p2_side", "p1_tuzdyk", "p2_tuzdyk"))
        assert state == want, (
            f"halfmove {row['halfmove']} ({expected}): state {state} != fixture {want}"
        )

        if row["winner"] != "ongoing":
            assert game.is_over, f"halfmove {row['halfmove']}: game should be over"
            expected_winner = {"P1": 0, "P2": 1, "draw": -1}[row["winner"]]
            assert game.winner() == expected_winner

    print(f"OK: replayed {len(rows)} halfmoves, all states match the 9Q fixture")

    # extra sanity: fresh game has exactly 9 legal moves, all distinct notations
    fresh = Game()
    moves = fresh.legal_moves()
    assert len(moves) == 9 and len({m.notation for m in moves}) == 9
    print("OK: initial position has 9 distinct legal moves")

    # tuzdyk scenario (not covered by the fixture): P1 sows 3 stones from
    # hole 9 -> last stone lands in P2's hole 2 (pit 10) making it 3 -> "92x"
    g = Game()
    g.pits = [9, 9, 9, 9, 9, 9, 9, 9, 3] + [2, 2, 9, 9, 9, 9, 9, 9, 9]
    move = g.play_notation("92x")
    assert move.makes_tuzdyk and g.tuzdyks[0] == 1 and g.kazans[0] == 3
    assert g.pits[10] == 0
    # P2 may not move from their own hole 2 (it is P1's tuzdyk now)
    assert 1 not in g.legal_actions()
    # P2 sows 3 stones from their hole 1 (pit 9): one falls into the tuzdyk
    # and must be collected by its owner P1
    g.play_notation("13")
    assert g.kazans[0] == 3 + 1, "stone sown into tuzdyk must go to owner"
    assert g.pits[10] == 0
    print("OK: tuzdyk creation, blocking, and collection behave correctly")

    # BOTH players may create one tuzdyk each (as on the real scoresheet:
    # 47x by White then 91x by Black) - but never a second one, never in
    # hole 9, and never mirroring the opponent's tuzdyk column.
    g = Game()
    g.tuzdyks = [4, -1]  # White already owns a tuzdyk (Black's hole 5)
    g.to_play = 1
    g.pits = [9, 9, 2, 9, 9, 9, 9, 9, 9] + [9] * 9
    # Black hole 4 (pit 12), 9 stones: lands (12+9-1)%18 = pit 2, making 3
    move = g.play_notation("43x")
    assert move.makes_tuzdyk and g.tuzdyks == [4, 2]
    # a second tuzdyk for the same player must be impossible anywhere
    g.to_play = 1
    assert not any(m.makes_tuzdyk for m in g.legal_moves())
    print("OK: both players can own one tuzdyk; no second tuzdyk possible")

    # mirror restriction: creation blocked where the opponent's tuzdyk sits
    g = Game()
    g.tuzdyks = [-1, 2]  # Black owns White's hole 3 (column index 2)
    g.pits = [9] * 18
    g.pits[8] = 4  # White hole 9: lands (8+4-1)%18 = pit 11 (Black col 2)
    g.pits[11] = 2  # becomes 3 on landing - but mirrored, so no tuzdyk
    move = g.play_notation("93")
    assert not move.makes_tuzdyk and g.tuzdyks[0] == -1

    # hole 9 restriction: landing makes 3 in opponent hole 9 -> no tuzdyk
    g = Game()
    g.pits = [9] * 18
    g.pits[8] = 10  # White hole 9: lands (8+10-1)%18 = pit 17 (Black hole 9)
    g.pits[17] = 2  # becomes 3 on landing
    move = g.play_notation("99")
    assert not move.makes_tuzdyk and g.tuzdyks[0] == -1, "hole 9 must never be tuzdyk"
    print("OK: mirror and hole-9 tuzdyk restrictions enforced")


if __name__ == "__main__":
    main()
