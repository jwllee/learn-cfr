"""
Implementation of simple version of Liar's Dices with 2 players and 1 6-sided die each.
"""

import numpy as np
import pandas as pd
import time, copy, math, os
from functools import reduce


from utils import make_logger


logger = make_logger('liar_dices_cfr.py')


N_DIE_SIDES = 6
WILD_CARD = N_DIE_SIDES
N_PLAYERS = 2
N_DICES = 1
N_ACTIONS = N_PLAYERS * N_DICES * N_DIE_SIDES + 1


#====================================================================================================
# Utility functions
#====================================================================================================


def infoset2int(player_roll, is_claimed):
    """Convert information set to hash number
    """
    infoset_num = reduce(lambda x, y: x * 10 + y, player_roll, 1)
    n_actions = len(is_claimed)
    # since we do not need to include the Dudo action as it should not be in an information set
    for action in range(n_actions - 2, -1, -1):
        infoset_num = 2 * infoset_num + int(is_claimed[action])

    return infoset_num


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
    dices = np.sort(dices, axis=0)
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


def get_last_claim(is_claimed):
    if not is_claimed.any():
        return None
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


def action2str(action, claim_num, claim_rank):
    num = claim_num[action]
    rank = claim_rank[action]
    return '{}-{}'.format(num, rank)


def strategy2str(strategy, claim_num, claim_rank):
    str_ = '['
    for action in range(N_ACTIONS - 1):
        if str_ != '[':
            str_ += ', '
        action_str = action2str(action, claim_num, claim_rank)
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
        action_str = action2str(action, claim_num, claim_rank)
        str_i = '{}: {:.5f}'.format(
            action_str, regret_sum[action] 
        )
        str_ += str_i
    str_ += ', {}: {:.5f}'.format('Dudo', regret_sum[N_ACTIONS - 1])
    str_ += ']'
    return str_


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
            else:
                # just assign uniform probability over all possible actions
                # since there is no counterfactual regrets
                self.strategy[action] = 1. / len(self.legal_actions)
            self.strategy_sum[action] += reach_prob * self.strategy[action]

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
        return np.concatenate((self.player_roll, self.is_claimed, self.regret_sum, self.get_average_strategy()))


class GameHistory:
    def __init__(self, n_actions, first_claimer=0):
        self.n_actions = n_actions
        self.claimer_history = [[first_claimer]]
        self.claim_history = [np.zeros(n_actions, dtype=np.bool)]

    def __len__(self):
        return len(self.claim_history)

    @property
    def cur_claimer_history(self):
        return self.claimer_history[-1]

    @property
    def cur_claim_history(self):
        return self.claim_history[-1]

    @property
    def round_no(self):
        return len(self.claim_history)

    @property
    def cur_player(self):
        return self.cur_claimer_history[-1]

    def get_next_player(self, active_players):
        n_active = active_players.sum()
        err_msg = 'Number of active players: {}'.format(n_active)
        assert n_active > 1, err_msg

        if len(self.cur_claimer_history) == 1:
            return self.cur_claimer_history[0]
        n_players = len(active_players)
        next_player = self.cur_player
        candidate = next_player
        while next_player == self.cur_player:
            candidate = (candidate + 1) % n_players
            if active_players[candidate]:
                next_player = candidate
                break

        return next_player

    def add_new_round(self, first_claimer=0):
        is_claimed = np.zeros(self.n_actions, dtype=np.bool)
        self.claim_history.append(is_claimed)
        self.claimer_history.append([first_claimer])

    def clone(self):
        cloned = GameHistory(self.n_actions)
        cloned.claim_history = copy.deepcopy(self.claim_history)
        cloned.claimer_history = copy.deepcopy(self.claimer_history)
        return cloned

    def __repr__(self):
        repr_ = '{}(current claimer history: {}, current claim history: {})'
        repr_ = repr_.format(GameHistory.__name__, 
                             self.cur_claimer_history, 
                             self.cur_claim_history)
        return repr_


class CFRInstance:
    def __init__(self, wild_card=WILD_CARD):
        self.claim_rank = get_claim_rank(N_PLAYERS, N_DICES, N_DIE_SIDES)
        self.claim_num = get_claim_num(N_PLAYERS, N_DICES, N_DIE_SIDES)
        self.infoset_map = dict()
        self.wild_card = wild_card

    def get_terminal_utility(self, history, dices):
        assert isinstance(history, GameHistory)

        n_actives = get_n_active_players(dices)

        winner = None
        utility = None

        if n_actives == 1:
            # using the fact that there are only 2 players, return the appropriate value
            # so that it can be flipped in the parent layer
            last_player = history.claimer_history[-2][-1]
            # would have been the opposite player's turn
            would_have_been = 1 - last_player
            winner = history.cur_player
            if would_have_been == winner:
                utility = 1
            else:
                utility = -1

            # if history.claim_history[-2][5]:
            #     info_msg = 'Last Player {}, Would have been: Player {}, Winner: Player {}, Utility: {}, Claim history: {}'
            #     info_msg = info_msg.format(last_player, would_have_been, winner, utility, history.claim_history[-2])
            #     logger.info(info_msg)

        return utility

    def get_infoset(self, infoset_id, player_roll, is_claimed):
        node = self.infoset_map.get(infoset_id)
        if node is None:
            node = InfoSetContainer(infoset_id, player_roll, is_claimed.copy(), self)
            self.infoset_map[infoset_id] = node
        else:
            assert (node.is_claimed == is_claimed).all()
            assert (node.player_roll == player_roll).all()
        return node

    def accumulate_regret(self, player, node, util, node_util, p0, p1, dices):
        """For each action, compute and accumulate counterfactual regret.
        """
        err_msg = '{!r} is not instance of {}'.format(node, InfoSetContainer.__name__)
        assert isinstance(node, InfoSetContainer), err_msg

        counterfactual_reach_prob = p1 if player == 0 else p0

        for action in node.legal_actions:
            regret = util[action] - node_util

            # if node.player_roll == 1 and node.is_claimed[4]:
            #     info_msg = "Player {}'s regret for not taking action {}: {}, claim history: {}, node_util: {}, util: {}, dices: {}"
            #     info_msg = info_msg.format(player, action, regret, node.is_claimed, node_util, util, dices.ravel())
            #     logger.info(info_msg)

            node.regret_sum[action] += counterfactual_reach_prob * regret

    def handle_round_end(self, history, dices):
        """ 
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

        if not history.cur_claim_history[-1]:
            # info_msg = 'Dudo action not taken in claim history: {}'
            # info_msg = info_msg.format(history.cur_claim_history)
            # logger.info(info_msg)
            return False

        # info_msg = 'Handling end of round {}'.format(history.round_no)
        # logger.info(info_msg)

        challenger = history.cur_claimer_history[-1]
        claimer = history.cur_claimer_history[-2]

        claim = np.argwhere(history.cur_claim_history).ravel()[-2]
        claimed_rank = self.claim_rank[claim]
        claimed_num = self.claim_num[claim]
        actual_num = get_num_of_rank(dices, claimed_rank, self.wild_card)
        diff = abs(actual_num - claimed_num)

        dices_copied = dices.copy()

        # Condition 1
        if actual_num > claimed_num:
            winner = claimer
            deactivate_player_dices(challenger, diff, dices)
        # Condition 2
        elif actual_num < claimed_num:
            winner = challenger
            deactivate_player_dices(claimer, diff, dices)
        # Condition 3
        else:
            winner = claimer
            for player in range(N_PLAYERS):
                if player == claimer:
                    continue
                deactivate_player_dices(player, 1, dices)

        # claim_str = '{}-{}'.format(claimed_num, claimed_rank)
        # info_msg = 'Claim: {}, Dices: {}, Claimer: Player {}, Challenger: Player {}, Winner: Player {}, Claim history: {}, Claimer history: {}'
        # info_msg = info_msg.format(
        #     claim_str, dices_copied.ravel(), 
        #     claimer, challenger, 
        #     winner, history.cur_claim_history,
        #     history.cur_claimer_history
        # )
        # logger.info(info_msg)
        # time.sleep(10)

        history.add_new_round(winner)
        roll_dices(dices, N_PLAYERS, N_DICES, N_DIE_SIDES)
        return True

    def cfr_action(self, node, history, dices, p0, p1):
        """For each action, recursively call cfr with additional history and probability.
        """
        assert isinstance(history, GameHistory)
        assert isinstance(node, InfoSetContainer)

        active_players = get_player_n_dices(dices) > 0
        player = history.get_next_player(active_players)
        reach_prob = p0 if player == 0 else p1

        err_msg = 'Reach probability of Player {}: {}, claim history: {}'
        err_msg = err_msg.format(player, reach_prob, history.cur_claim_history)
        assert reach_prob > 0, err_msg

        strategy = node.get_strategy(reach_prob)
        util = np.zeros(N_ACTIONS)
        node_util = 0.

        for action in node.legal_actions:

            # we will never visit this subgame in real life
            if strategy[action] <= 0.:
                continue

            history_copied = history.clone()
            dices_copied = dices.copy()

            history_copied.cur_claimer_history.append(player)
            history_copied.cur_claim_history[action] = True

            player_copied = history_copied.get_next_player(active_players)
            assert player_copied != player

            # err_msg = 'Legal action {} probability <= 0: {}, reach prob: {}, node: {}, strategy: {}'
            # err_msg = err_msg.format(action, strategy[action], reach_prob, node, strategy)
            # assert strategy[action] > 0., err_msg

            # info_msg = 'Modified with action {}, Copied claim history: {}'
            # info_msg = info_msg.format(action, history_copied.cur_claim_history)
            # logger.info(info_msg)
            # time.sleep(10)
            # if node.player_roll == 1 and node.is_claimed[5]:
            #     info_msg = 'Player {} check utility of action {}, dices: {}'
            #     info_msg = info_msg.format(player, action, dices.ravel())
            #     logger.info(info_msg)

            if player == 0:
                p0_action = p0 * strategy[action]
                util[action] = -self.cfr(history_copied, dices_copied, p0_action, p1)
            else:
                p1_action = p1 * strategy[action]
                util[action] = -self.cfr(history_copied, dices_copied, p0, p1_action)

            # if node.player_roll == 1 and node.is_claimed[5]:
            #     info_msg = 'Player {}, Utility of action {}: {}, Claim history: {}'
            #     info_msg = info_msg.format(player, action, util[action], history.cur_claim_history)
            #     logger.info(info_msg)

            node_util += strategy[action] * util[action]

        return util, node_util

    def cfr(self, history, dices, p0, p1):
        assert isinstance(history, GameHistory)

        # info_msg = 'cfr: claim history: {}'
        # info_msg = info_msg.format(history.cur_claim_history)
        # logger.info(info_msg)
        # time.sleep(10)

        round_ended = self.handle_round_end(history, dices)

        utility = self.get_terminal_utility(history, dices)

        if utility is not None:
            return utility

        active_players = get_player_n_dices(dices) > 0
        player = history.get_next_player(active_players)
        player_roll = dices[player]
        infoset_id = infoset2int(player_roll, history.cur_claim_history)

        # Get information set node or create it if nonexistent
        node = self.get_infoset(infoset_id, player_roll, history.cur_claim_history)

        # For each action, recursively call cfr with additional history and probability
        util, node_util = self.cfr_action(node, history, dices, p0, p1)
        
        # For each action, compute and accumulate counterfactual regret
        self.accumulate_regret(player, node, util, node_util, p0, p1, dices)

        # info_msg = 'Infoset {}, regret sum: {}'
        # info_msg = info_msg.format(node, node.regret_sum)
        # logger.info(info_msg)

        return node_util

    def train(self, it):
        util = 0
        for i in range(it):
            if (i + 1) % 1000 == 0:
                info_msg = 'Running iteration {}, Current average game value: {}, {} info sets'
                info_msg = info_msg.format(i + 1, util / i, len(self.infoset_map))
                logger.info(info_msg)

            dices = get_dices(N_PLAYERS, N_DICES, N_DIE_SIDES)
            history = GameHistory(N_ACTIONS)

            util_i = self.cfr(history, dices, 1, 1)
            util += util_i

        print()
        info_msg = 'Average game value: {:.4f}'.format(util / it)
        logger.info(info_msg)

        print()
        info_msg = '{} information sets'.format(len(self.infoset_map))
        logger.info(info_msg)

        return self.infoset_map.values()

#             assert isinstance(node, InfoSetContainer)
#             if node.player_roll == 3:
#                 count += 1
#                 nodes.append(node)
#         nodes = sorted(nodes, key=lambda n: is_claimed_key(n.is_claimed))
#         for node in nodes:
#             print(node)
#         logger.info('{} infosets'.format(count))
 

if __name__ == '__main__':
    # dices = get_dices(N_PLAYERS + 2, N_DICES + 5, N_DIE_SIDES)
    # print('Dices: \n{}'.format(dices))

    # n_dices = get_n_dices(dices[0])
    # print('No. of dices in {}: {}'.format(dices[0], n_dices))

    # deactivate_player_dices(2, 3, dices)
    # print('Dices: \n{}'.format(dices))

    # player_n_dices = get_player_n_dices(dices)
    # print('No. of player dices: {}'.format(player_n_dices))

    it = 1000
    start = time.time()
    instance = CFRInstance()
    infosets = instance.train(it)

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

    df = pd.DataFrame.from_records(infoset_rows, columns=header)
    df['player_roll'] = df['player_roll'].astype(int)
    df[is_claimed_claims] = df[is_claimed_claims].astype(bool)
    df.to_csv(out_fp, index=None, float_format='%.8f')

