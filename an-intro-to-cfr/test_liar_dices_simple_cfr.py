import pytest
import numpy as np
import liar_dices_simple_cfr as ld


def test_get_claim_num():
    n_players, n_dices, n_die_sides = 2, 1, 6
    expected_claim_num = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 ]
    claim_num = ld.get_claim_num(n_players, n_dices, n_die_sides)
    assert (claim_num == expected_claim_num).all()


def test_get_claim_rank():
    n_players, n_dices, n_die_sides = 2, 1, 6
    expected_claim_rank = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    claim_rank = ld.get_claim_rank(n_players, n_dices, n_die_sides)
    assert (claim_rank == expected_claim_rank).all()


def test_get_num_of_rank():
    n_players, n_dices = 2, 1
    wild_card = 6

    # special case
    dices = np.asarray([3, 6]).reshape((n_players, n_dices))
    rank, expected_count = 3, 2
    count = ld.get_num_of_rank(dices, rank, wild_card)
    assert count == expected_count 

    rank, expected_count = 6, 1
    count = ld.get_num_of_rank(dices, rank, wild_card)
    assert count == expected_count


def test_get():
    n_players, n_dices, n_die_sides = 2, 1, 6
    n_actions = n_players * n_dices * n_die_sides + 1
    is_claimed = np.zeros(n_actions, dtype=np.bool)
    player_roll_1 = [1]
    player_roll_2 = [2]
    player_roll_3 = [3]
    player_roll_5 = [5]
    player_roll_6 = [6]

    num_1 = ld.infoset2int(player_roll_1, is_claimed)
    num_2 = ld.infoset2int(player_roll_2, is_claimed)
    num_3 = ld.infoset2int(player_roll_3, is_claimed)
    num_5 = ld.infoset2int(player_roll_5, is_claimed)
    num_6 = ld.infoset2int(player_roll_6, is_claimed)

    assert num_1 != num_5
    assert num_1 != num_6
    assert num_2 != num_5
    assert num_2 != num_6
    assert num_3 != num_5
    assert num_3 != num_6

    # try different is_claimed


class TestGameHistory:
    def test_initial_state(self):
        n_players, n_dices, n_die_sides = 2, 1, 6
        n_actions = n_players * n_dices * n_die_sides + 1
        player_0, player_1 = 0, 1
        gh = ld.GameHistory(n_actions)

        assert gh.cur_player == player_0
        assert gh.is_terminated() == False
        assert gh.get_next_player() == player_1
        assert gh.claim_history.any() == False

        cloned = gh.clone()
        assert cloned.cur_player == player_0
        assert cloned.is_terminated() == False
        assert cloned.get_next_player() == player_1
        assert cloned.claim_history.any() == False

        claim = 0
        cloned.update(claim)
        assert cloned.cur_player == player_1
        assert cloned.is_terminated() == False
        assert cloned.get_next_player() == player_0
        assert cloned.claim_history.any() == True
        assert cloned.claim_history[claim] == True

        # make sure original history does not get changed
        assert gh.cur_player == player_0
        assert gh.is_terminated() == False
        assert gh.get_next_player() == player_1
        assert gh.claim_history.any() == False

    def test_update(self):
        n_players, n_dices, n_die_sides = 2, 1, 6
        n_actions = n_players * n_dices * n_die_sides + 1
        player_0, player_1 = 0, 1
        gh = ld.GameHistory(n_actions)

        claim = 1
        gh.update(claim)

        claim = n_actions - 1
        gh.update(claim)

        assert gh.is_terminated() == True
        assert gh.cur_player == player_0


class TestCFRInstance:
    def test_get_terminal_utility_player_1_dudo(self):
        n_players, n_dices, n_die_sides = 2, 1, 6
        n_actions = n_players * n_dices * n_die_sides + 1
        player_0, player_1 = 0, 1
        cfr = ld.CFRInstance()
        history = ld.GameHistory(n_actions)

        dices = np.asarray([3, 6]).reshape((n_players, n_dices))
        assert cfr.wild_card == ld.WILD_CARD
        utility = cfr.get_terminal_utility(player_0, history, dices)
        assert utility is None

        claim = 1
        history.update(claim)

        claim = n_actions - 1
        history.update(claim)

        assert history.cur_player == player_0
        assert history.is_terminated() == True

        # claim is True
        dices = np.asarray([3, 6]).reshape((n_players, n_dices))
        utility_0 = cfr.get_terminal_utility(player_0, history, dices)
        utility_1 = cfr.get_terminal_utility(player_1, history, dices)
        assert utility_0 == 1
        assert utility_1 == -1

        # claim is False
        dices = np.asarray([3, 5]).reshape((n_players, n_dices))
        utility_0 = cfr.get_terminal_utility(player_0, history, dices)
        utility_1 = cfr.get_terminal_utility(player_1, history, dices)
        info_msg = 'Claim history: {}, dices: {}'
        info_msg = info_msg.format(history.claim_history, dices.ravel())
        assert utility_0 == -1, info_msg
        assert utility_1 == 1

    def test_get_terminal_utility_player_0_dudo(self):
        n_players, n_dices, n_die_sides = 2, 1, 6
        n_actions = n_players * n_dices * n_die_sides + 1
        player_0, player_1 = 0, 1
        cfr = ld.CFRInstance()
        history = ld.GameHistory(n_actions)
        dices = np.asarray([3, 6]).reshape((n_players, n_dices))

        assert cfr.wild_card == ld.WILD_CARD
        utility = cfr.get_terminal_utility(player_0, history, dices)
        assert utility is None

        claim = 1
        history.update(claim)

        claim = 8
        history.update(claim)

        claim = n_actions - 1
        history.update(claim)

        assert history.cur_player == player_1
        assert history.is_terminated() == True

        # claim is True
        dices = np.asarray([3, 6]).reshape((n_players, n_dices))
        utility_0 = cfr.get_terminal_utility(player_0, history, dices)
        utility_1 = cfr.get_terminal_utility(player_1, history, dices)

        assert utility_0 == -1
        assert utility_1 == 1

        # claim is False
        dices = np.asarray([1, 3]).reshape((n_players, n_dices))
        utility_0 = cfr.get_terminal_utility(player_0, history, dices)
        utility_1 = cfr.get_terminal_utility(player_1, history, dices)

        assert utility_0 == 1
        assert utility_1 == -1
