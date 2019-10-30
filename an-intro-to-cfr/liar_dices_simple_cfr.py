"""
Implementation of simple version of Liar's dices with 2 players and 1 6-sided die each.
"""

import numpy as np
import pandas as pd
import time, copy, math, os
from functools import reduce
import multiprocessing as mp


from utils import make_logger


logger = make_logger('liar_dices_cfr.py')


N_DIE_SIDES = 6
WILD_CARD = 6
N_PLAYERS = 2
N_DICES = 1
N_ACTIONS = N_PLAYERS * N_DICES * N_DIE_SIDES + 1


#====================================================================================================
# Utility functions
#====================================================================================================


def infoset2int(player_roll, is_claimed):
    """Convert information set to hash number
    """
    infoset_num = player_roll[0]
    n_actions = len(is_claimed)
    # since we do not need to include the Dudo action as it should not be in an information set
    for action in range(n_actions - 2, -1, -1):
        infoset_num = 2 * infoset_num + int(is_claimed[action])

    return infoset_num


def infoset2str(player_roll, is_claimed):
    str_ = str(player_roll[0]) + '|'
    n_actions = len(is_claimed)
    for action in range(0, n_actions - 1):
        str_ = str_ + str(int(is_claimed[action]))
    return str_


def claim_history2str(is_claimed, claim_num, claim_rank):
    n_actions = len(is_claimed)
    str_ = ''
    for action in range(n_actions - 1):
        if is_claimed[action]:
            if str_ != '':
                str_.append(',')
            str_.append(claim_num[action])
            str_.append('-')
            str_.append(claim_rank[action])

    # check for dudo bid
    if is_claimed[n_action - 1]:
        # str_ cannot be ''
        err_msg = 'Cannot have dudo without previous bids: {}'.format(is_claimed)
        assert str_ != '', err_msg
        str_.append(',dudo')

    return str_


def claim2str(claim, claim_num, claim_rank):
    num = claim_num[claim]
    rank = claim_rank[claim]
    return '{}-{}'.format(num, rank)


def strategy2str(strategy, claim_num, claim_rank):
    str_ = '['
    for action in range(N_ACTIONS - 1):
        if str_ != '[':
            str_ += ', '
        action_str = claim2str(action, claim_num, claim_rank)
        str_i = '{}: {:.3f}%'.format(
            action_str, strategy[action] * 100.
        )
        str_ += str_i
    str_ += ', {}: {:.3f}%'.format('Dudo', strategy[N_ACTIONS - 1] * 100.)
    str_ += ']'
    return str_


def regret_sum2str(regret_sum, claim_num, claim_rank):
    str_ = '['
    for action in range(N_ACTIONS - 1):
        if str_ != '[':
            str_ += ', '
        action_str = claim2str(action, claim_num, claim_rank)
        str_i = '{}: {:.5f}'.format(
            action_str, regret_sum[action] 
        )
        str_ += str_i
    str_ += ', {}: {:.5f}'.format('Dudo', regret_sum[N_ACTIONS - 1])
    str_ += ']'
    return str_


def get_claim_num(n_players, n_dices, n_die_sides):
    claim_num = []
    for i in range(n_players * n_dices):
        claim_num += [i + 1 for _ in range(n_die_sides)]
    claim_num = np.asarray(claim_num)
    return claim_num


def get_claim_rank(n_players, n_dices, n_die_sides):
    claim_rank = []
    for i in range(n_players * n_dices):
        claim_rank += [i + 1 for i in range(n_die_sides)]
    claim_rank = np.asarray(claim_rank)
    return claim_rank


def get_dices(n_players, n_dices, n_die_sides):
    low, high = 1, n_die_sides + 1
    shape = (n_players, n_dices)
    dices = np.random.randint(low, high, shape)
    return dices


def roll_dices(dices, n_players, n_dices, n_die_sides):
    mask = dices > 0
    # places where there are zero dices will remain with zero dices
    dices[:] = mask * get_dices(n_players, n_dices, n_die_sides)


def get_n_dices(player_roll):
    return (player_roll > 0).sum()


def get_num_of_rank(dices, rank, wild_card):
    raveled = np.ravel(dices)
    count = 0
    for die in raveled:
        count += (die == rank) or (die == wild_card)
    return count


def get_player_n_dices(dices):
    return np.apply_along_axis(get_n_dices, axis=1, arr=dices)


def get_n_active_players(dices):
    player_n_dices = get_player_n_dices(dices)
    return (player_n_dices > 0).sum()


def deactivate_player_dices(player, n, dices):
    """Assumes that dices have been sorted ascending order per player and 
    that the corresponding player has dices to be deactivated.
    """
    n_dices = dices.shape[1]
    start = np.argwhere(dices[player] > 0).ravel()[0]
    if start + n <= n_dices:
        end = start + n
    else:
        end = n_dices
    dices[player, start:end] = 0


def get_last_claim(is_claimed):
    if not is_claimed.any():
        err_msg = 'No claims in is_claimed: {}'
        err_msg = err_msg.format(is_claimed)
        raise ValueError(err_msg)
    return np.argwhere(is_claimed).ravel()[-1]


def get_legal_actions(is_claimed):
    if not is_claimed.any():
        legal_actions = list(range(N_ACTIONS - 1))
    else:
        last_claim = get_last_claim(is_claimed)
        legal_actions = list(range(last_claim + 1, N_ACTIONS))

    # info_msg = 'is_claimed: {}, legal_actions: {}'
    # info_msg = info_msg.format(is_claimed, legal_actions)
    # logger.info(info_msg)

    return legal_actions


def is_terminated(is_claimed):
    return is_claimed[-1]


def check_is_ongoing(is_claimed):
    if is_terminated(is_claimed):
        err_msg = 'Game is terminated with claim history: {}'
        err_msg = err_msg.format(is_claimed)
        raise ValueError(err_msg)


def check_claim_is_valid(claim, is_claimed):
    # no claims have been made
    if not is_claimed.any():
        if claim == len(is_claimed) - 1:
            err_msg = 'Dudo cannot be taken when no other claims have been done.'
            raise ValueError(err_msg)
        return 

    last_claim = get_last_claim(is_claimed)
    if claim <= last_claim or claim < 0:
        err_msg = 'Claim {} is not valid given is_claimed: {}'
        err_msg = err_msg.format(claim, is_claimed)
        raise ValueError(err_msg)


#====================================================================================================


class InfoSetContainer:
    def __init__(self, infoset_id, player_roll, is_claimed, game):
        self.game = game
        self.player_roll = player_roll
        self.infoset_id = infoset_id
        self.is_claimed = is_claimed
        self.legal_actions = get_legal_actions(is_claimed)

        self.regret_sum = np.zeros(N_ACTIONS)
        self.strategy = np.zeros(N_ACTIONS)
        self.strategy_sum = np.zeros(N_ACTIONS)

    def get_strategy(self, reach_prob):
        normalizing_sum = 0.

        for action in self.legal_actions:
            if self.regret_sum[action] > 0:
                self.strategy[action] = self.regret_sum[action]
            else:
                self.strategy[action] = 0.
            normalizing_sum += self.strategy[action]

        for action in self.legal_actions:
            if normalizing_sum > 0:
                self.strategy[action] /= normalizing_sum

                if self.strategy[action] < 0.002:
                    self.strategy[action] = 0.

            else:
                # just assign uniform probability over all possible actions
                # since there is no counterfactual regrets
                self.strategy[action] = 1. / len(self.legal_actions)
            self.strategy_sum[action] += reach_prob * self.strategy[action]

        if normalizing_sum > 0:
            self.strategy = self.strategy / self.strategy.sum()

        strategy_sum = self.strategy.sum()
        err_msg = 'Strategy {} sums to {:.2f}'.format(self.strategy, strategy_sum)
        assert math.isclose(strategy_sum, 1) is True, err_msg

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = np.zeros(N_ACTIONS)
        normalizing_sum = self.strategy_sum[self.legal_actions].sum()

        if normalizing_sum > 0.:
            avg_strategy[self.legal_actions] = self.strategy_sum[self.legal_actions] / normalizing_sum
        else:
            avg_strategy[self.legal_actions] = 1. / len(self.legal_actions)

        strategy_sum = avg_strategy.sum()
        err_msg = 'Average strategy {} sums to {:.2f}'.format(avg_strategy, strategy_sum)
        assert math.isclose(strategy_sum, 1) is True, err_msg

        return avg_strategy

    def __repr__(self):
        repr_ = '{} Roll: {}, Claimed: {}, Strategy: {}, Regret sum: {}'
        avg_strategy = self.get_average_strategy()
        regret_sum_str = regret_sum2str(self.regret_sum, 
                                      self.game.claim_num,
                                      self.game.claim_rank)
        avg_strategy_str = strategy2str(avg_strategy, 
                                        self.game.claim_num, 
                                        self.game.claim_rank)

        strategy_sum = avg_strategy.sum()
        err_msg = 'Average strategy {} sums to {:.2f}'.format(avg_strategy, strategy_sum)
        assert math.isclose(strategy_sum, 1) is True, err_msg

        repr_ = repr_.format(
            InfoSetContainer.__name__,
            self.player_roll,
            self.is_claimed,
            avg_strategy_str,
            regret_sum_str,
        )
        return repr_

    def __str__(self):
        return str(repr(self))

    def to_row(self):
        to_concat = [
            self.player_roll,
            self.is_claimed,
            self.regret_sum,
            self.get_average_strategy()
        ]
        return np.concatenate(to_concat)


class GameHistory:
    """Game history for an one round game.
    """
    def __init__(self, n_actions, first_claimer=0):
        self.n_actions = n_actions
        self.claim_history = np.zeros(n_actions, dtype=np.bool)
        self.cur_player = first_claimer

    def is_terminated(self):
        # game is terminated when someone calls Dudo
        return is_terminated(self.claim_history)

    def get_next_player(self):
        check_is_ongoing(self.claim_history)
        return 1 - self.cur_player

    def clone(self):
        cloned = GameHistory(self.n_actions)
        cloned.claim_history = copy.deepcopy(self.claim_history)
        cloned.cur_player = self.cur_player
        return cloned

    def update(self, claim):
        check_is_ongoing(self.claim_history)
        check_claim_is_valid(claim, self.claim_history)
        aux = self.cur_player
        self.cur_player = self.get_next_player()
        self.claim_history[claim] = True

    def __repr__(self):
        repr_ = '{}(current claim history: {})'
        repr_ = repr_.format(GameHistory.__name__, 
                             self.claim_history)
        return repr_


class CFRInstance:
    def __init__(self, wild_card=WILD_CARD):
        self.claim_rank = get_claim_rank(N_PLAYERS, N_DICES, N_DIE_SIDES)
        self.claim_num = get_claim_num(N_PLAYERS, N_DICES, N_DIE_SIDES)
        self.infoset_map = dict()
        self.wild_card = wild_card

    def get_infoset(self, infoset_id, player_roll, is_claimed):
        node = self.infoset_map.get(infoset_id)
        if node is None:
            node = InfoSetContainer(infoset_id, player_roll, is_claimed.copy(), self)
            self.infoset_map[infoset_id] = node
        assert (node.is_claimed == is_claimed).all()
        assert node.player_roll == player_roll
        return node

    def get_terminal_utility(self, player, history, dices):
        """Get terminal state utility for player if possible.

        Check for the three possible conditions
        1. Actual rank count exceeds the challenged claim:
        Challenger loses a number of dices equal to the difference between the 
        actual rank count and the claim count.
        
        2. Actual rank count is less than the challenged claim
        The challenged player loses a number of dices equal to the difference 
        between the claim count and the actual rank count.
        
        3. Actual rank count is equal to the challenged claim
        Every player except the challenged player loses a single die. 
        """
        assert isinstance(history, GameHistory)

        if not history.is_terminated():
            return None

        winner, utility = None, None

        # we are now at terminal node, so there should not be a current
        # player, instead it was the player before the current player who
        # challenged the claim.
        challenger = 1 - history.cur_player
        claimer = 1 - challenger
        
        # second last claim, i.e., last claim excluding the dudo claim
        claim = get_last_claim(history.claim_history[:-1])
        claimed_rank = self.claim_rank[claim]
        claimed_num = self.claim_num[claim]
        actual_num = get_num_of_rank(dices, claimed_rank, self.wild_card)
        diff = abs(actual_num - claimed_num)

        # Condition 1
        if actual_num > claimed_num:
            winner = claimer
        # Condition 2
        elif actual_num < claimed_num:
            winner = challenger
        # Condition 3
        else:
            winner = claimer

        if winner == player:
            utility = 1
        else:
            utility = -1

        # claim_str = claim2str(claim, self.claim_num, self.claim_rank)
        # info_msg = 'Winner: Player {}, Claim: {}, Dices: {}'
        # info_msg = info_msg.format(winner, claim_str, dices.ravel())
        # logger.info(info_msg)

        # info_msg = 'Claimer: Player {}, Challenger: Player {}, Claim: {}'
        # info_msg = info_msg.format(claimer, challenger, claim_str)
        # logger.info(info_msg)
        # info_msg = 'Utility for Player {}: {}, claim history: {}, dices: {}'
        # info_msg = info_msg.format(player, utility, history.claim_history, dices.ravel())
        # logger.info(info_msg)

        return utility

    def accumulate_regret(self, player, node, util, node_util, p0, p1, dices):
        """For each action, compute and accumulate counterfactual regret.
        """
        err_msg = '{!r} is not instance of {}'.format(node, InfoSetContainer.__name__)
        assert isinstance(node, InfoSetContainer), err_msg

        counterfactual_reach_prob = p1 if player == 0 else p0

        for action in node.legal_actions:
            regret = util[action] - node_util
            node.regret_sum[action] += counterfactual_reach_prob * regret

    def cfr_action(self, node, history, dices, p0, p1):
        """For each action, recursively call cfr with additional history and probability.
        """
        assert isinstance(history, GameHistory)
        assert isinstance(node, InfoSetContainer)

        player = history.cur_player
        reach_prob = p0 if player == 0 else p1

        err_msg = 'Reach probability of Player {}: {}, claim history: {}'
        err_msg = err_msg.format(player, reach_prob, history.claim_history)
        assert reach_prob > 0, err_msg

        strategy = node.get_strategy(reach_prob)
        util = np.zeros(N_ACTIONS)
        node_util = 0.

        # np.random.shuffle(node.legal_actions)
        for action in node.legal_actions:

            # we will never visit this subgame in real life
            if strategy[action] <= 0.:
                continue

            history_copied = history.clone()
            dices_copied = dices.copy()
            history_copied.update(action)

            if action == N_ACTIONS - 1:
                # this is dudo action
                util[action] = self.get_terminal_utility(player, history_copied, dices)
                # info_msg = 'Player {} has utility {} for claim history {}, dices: {}'
                # info_msg = info_msg.format(player, util, history_copied.claim_history, dices.ravel())
                # logger.info(info_msg)
                # time.sleep(10)
            else:
                # recursive call
                if player == 0:
                    p0_action = p0 * strategy[action]
                    util[action] = -self.cfr(history_copied, dices_copied, p0_action, p1)
                else:
                    p1_action = p1 * strategy[action]
                    util[action] = -self.cfr(history_copied, dices_copied, p0, p1_action)

            node_util += strategy[action] * util[action]

        return util, node_util

    def cfr(self, history, dices, p0, p1):
        assert isinstance(history, GameHistory)

        player = history.cur_player

        player_roll = dices[player]
        # infoset_id = infoset2int(player_roll, history.claim_history)
        infoset_id = infoset2str(player_roll, history.claim_history)

        # Get information set node or create it if nonexistent
        node = self.get_infoset(infoset_id, player_roll, history.claim_history)

        # For each action, recursively call cfr with additional history and probability
        util, node_util = self.cfr_action(node, history, dices, p0, p1)
        
        # For each action, compute and accumulate counterfactual regret
        self.accumulate_regret(player, node, util, node_util, p0, p1, dices)

        return node_util

    def train(self, it, reset_sum_it=-1):
        util = 0
        dice_count_0 = np.zeros(7)
        dice_count_1 = np.zeros(7)
        util_list = []
        for i in range(it):
            if (i + 1) % 1000 == 0:
                info_msg = 'Running iteration {}, Current average game value: {:.4f}, {} info sets'
                info_msg = info_msg.format(i + 1, util / i, len(self.infoset_map))
                logger.info(info_msg)

            # do not use chance sampling, just iterate through all possible dice rolls
            # dices = get_dices(N_PLAYERS, N_DICES, N_DIE_SIDES)
            util_i, n_combo = 0, 0

            for player_0_roll in range(1, N_DIE_SIDES + 1):
                for player_1_roll in range(1, N_DIE_SIDES + 1):
                    dices = np.asarray([player_0_roll, player_1_roll])
                    dices = dices.reshape((N_PLAYERS, N_DICES))
                    dice_count_0[dices[0,0]] += 1
                    dice_count_1[dices[1,0]] += 1
                    history = GameHistory(N_ACTIONS, first_claimer=0)

                    util_i += self.cfr(history, dices, 1, 1)
                    n_combo += 1

            # get the average utility over all possible combination, each combination 
            # should have the same probability to show up
            util_i = util_i / n_combo
            util += util_i

            if i == reset_sum_it:
                for infoset in self.infoset_map.values():
                    infoset.strategy_sum[:] = 0.

            if (i + 1) % 100 == 0:
                util_list.append((i, util / i))

        print()
        info_msg = 'Average game value: {:.4f}'.format(util / i)
        logger.info(info_msg)

        print()
        info_msg = '{} information sets'.format(len(self.infoset_map))
        logger.info(info_msg)

        print()
        info_msg = 'Player 0 dice count: {}'
        info_msg = info_msg.format(dice_count_0)
        logger.info(info_msg)

        info_msg = 'Player 1 dice count: {}'
        info_msg = info_msg.format(dice_count_1)
        logger.info(info_msg)

        return self.infoset_map.values(), util_list


if __name__ == '__main__':
    it = 1000
    reset_sum_it = 10
    start = time.time()
    instance = CFRInstance()
    infosets, util_list = instance.train(it, reset_sum_it=reset_sum_it)

    end = time.time()
    took = end - start
    logger.info('Took {:.2f}s'.format(took))

    infoset_rows = [s.to_row() for s in infosets]

    claim_num = get_claim_num(N_PLAYERS, N_DICES, N_DIE_SIDES)
    claim_rank = get_claim_rank(N_PLAYERS, N_DICES, N_DIE_SIDES)
    claims = []

    for action in range(N_ACTIONS - 1):
        claim = '{}-{}'.format(claim_num[action], claim_rank[action])
        claims.append(claim)
    claims.append('Dudo')

    is_claimed_claims = ['{}_{}'.format('is_claimed', c) for c in claims]
    regret_claims = ['{}_{}'.format('regret', c) for c in claims]
    avg_strategy_claims = ['{}_{}'.format('avg_strategy', c) for c in claims]

    header = [ 'player_roll' ] + is_claimed_claims + regret_claims + avg_strategy_claims
    out_fp = 'n_players_{}-n_dices_{}-n_die_sides_{}-it_{}.csv'
    out_fp = out_fp.format(N_PLAYERS, N_DICES, N_DIE_SIDES, it)

    out_dir = 'results'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out_fp = os.path.join(out_dir, out_fp)

    util_fp = 'n_players_{}-n_dices_{}-n_die_sides_{}-it_{}-util.csv'
    util_fp = util_fp.format(N_PLAYERS, N_DICES, N_DIE_SIDES, it)
    util_fp = os.path.join(out_dir, util_fp)
    util_df = pd.DataFrame.from_records(util_list, columns=['iteration', 'avg_util'])
    util_df.to_csv(util_fp, index=None)

    df = pd.DataFrame.from_records(infoset_rows, columns=header)
    df['player_roll'] = df['player_roll'].astype(int)
    df[is_claimed_claims] = df[is_claimed_claims].astype(bool)
    df.to_csv(out_fp, index=None, float_format='%.8f')
