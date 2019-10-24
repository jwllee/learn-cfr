"""
Implementation of counterfactual regret minimization (CFR) with chance sampling for Kuhn poker as shown in "An Introduction to Counterfactual Regret Minimization".
"""
import numpy as np
from utils import make_logger

np.set_printoptions(precision=3)


logger = make_logger('kp_cfr.py')


NB_ACTIONS = 2
PASS, BET = 0, 1 
PASS_STR, BET_STR = 'p', 'b'


class InfoSetContainer:
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = np.zeros(NB_ACTIONS)
        self.strategy = np.zeros(NB_ACTIONS)
        self.strategy_sum = np.zeros(NB_ACTIONS)

    def get_strategy(self, reach_prob):
        """
        :param reach_prob float: Probability of reaching the information set, this can be different at different times since there are multiple histories within an information set.
        """
        normalizing_sum = 0.

        for action in range(NB_ACTIONS):
            if self.regret_sum[action] > 0:
                self.strategy[action] = self.regret_sum[action] 
            else:
                self.strategy[action] = 0
            normalizing_sum += self.strategy[action]

        for action in range(NB_ACTIONS):
            if normalizing_sum > 0:
                self.strategy[action] /= normalizing_sum
            else:
                # just assign uniform probability over all possible actions
                # since there is no counterfactual regrets
                self.strategy[action] = 1. / NB_ACTIONS
            # need to incorporate the reach probability to this particular
            # history in the accumulation for the overall strategy
            # info_msg = 'strategy sum: {}, strategy: {}, reach_prob: {}'
            # info_msg = info_msg.format(self.strategy_sum, self.strategy, reach_prob)
            # logger.info(info_msg)

            self.strategy_sum[action] += reach_prob * self.strategy[action]

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = np.zeros(NB_ACTIONS)

        normalizing_sum = 0.
        for action in range(NB_ACTIONS):
            normalizing_sum += self.strategy_sum[action]

        for action in range(NB_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[action] = self.strategy_sum[action] / normalizing_sum
            else:
                avg_strategy[action] = 1. / NB_ACTIONS

        return avg_strategy

    def __repr__(self):
        repr_ = '{}: {}'
        avg_strategy = self.get_average_strategy()
        avg_strategy_str = '[Pass: {:.2f}%, Bet: {:.2f}%]'
        avg_strategy_str = avg_strategy_str.format(avg_strategy[0] * 100., 
                                                   avg_strategy[1] * 100.)
        repr_ = repr_.format(self.info_set, avg_strategy_str)
        return repr_

    def __str__(self):
        return str(repr(self))


def shuffle(cards):
    """Shuffle cards in place using the Durstenfeld version of the Fisher-Yates shuffle
    """
    for c1 in range(len(cards) - 1, -1, -1):
        c2 = np.random.randint(0, c1 + 1)
        tmp = cards[c1]
        cards[c1] = cards[c2]
        cards[c2] = tmp


class CFRInstance:
    def __init__(self):
        self.node_map = dict()

    def get_terminal_utility(self, player, opponent, plays, history, cards):
        if plays > 1:
            terminal_pass = history[plays - 1] == PASS_STR
            double_bet_str = '{}{}'.format(BET_STR, BET_STR)
            double_pass_str = '{}{}'.format(PASS_STR, PASS_STR)

            start, end = int(plays - 2), int(plays)
            double_bet = history[start:end] == double_bet_str
            is_player_card_higher = cards[player] > cards[opponent]
            winner = None
            utility = None

            if terminal_pass:
                if history == double_pass_str:
                    if is_player_card_higher:
                        winner = player
                        utility = 1
                    else:
                        winner = opponent
                        utility = -1
                else:
                    winner = player
                    utility = 1
            elif double_bet:
                if is_player_card_higher:
                    winner = player
                    utility = 2
                else:
                    winner = opponent
                    utility = -2

            # info_msg = 'Winner: Player {}, Uility: {}, History: {}'
            # info_msg = info_msg.format(winner, utility, history)
            # logger.info(info_msg)

            return utility

    def get_info_set(self, info_set):
        node = self.node_map.get(info_set)
        if node is None:
            node = InfoSetContainer(info_set)
            self.node_map[info_set] = node
        return node

    def cfr_action(self, player, node, history, p0, p1, cards):
        reach_prob = p0 if player == 0 else p1
        strategy = node.get_strategy(reach_prob)
        util = np.zeros(NB_ACTIONS)
        node_util = 0.

        for action in range(NB_ACTIONS):
            action_str = PASS_STR if action == PASS else BET_STR
            next_history = history + action_str

            if player == 0:
                p0_action = p0 * strategy[action]
                # You need to flip the sign since the CFR value is returned 
                # in the perspective of your opponent.
                cfr = -self.cfr(next_history, p0_action, p1, cards)
                util[action] = cfr
            else:
                p1_action = p1 * strategy[action]
                cfr = -self.cfr(next_history, p0, p1_action, cards)
                util[action] = cfr

            # info_msg = 'CFR for Player {} History {}: {}'
            # info_msg = info_msg.format(player, next_history, cfr)
            # logger.info(info_msg)

            node_util += strategy[action] * util[action]

        return util, node_util

    def accumulate_regret(self, player, node, util, node_util, p0, p1):
        """For each action, compute and accumulate counterfactual regret.
        
        :param node InfoSetContainer: information set container
        :param util array_like: counterfactual utility of actions 
        :param node_util float: counterfactual utility of information set container
        """
        err_msg = '{!r} is not instance of {}'.format(node, InfoSetContainer.__name__)
        assert isinstance(node, InfoSetContainer), err_msg

        for action in range(NB_ACTIONS):
            regret = util[action] - node_util
            counterfactual_reach_prob = p1 if player == 0 else p0
            node.regret_sum[action] += counterfactual_reach_prob * regret

    def cfr(self, history, p0, p1, cards):
        """
        :param history str: history of game play
        :param p0 float: reach probability of player 0
        :param p1 float: reach probability of player 1
        :param cards array_like: shuffled cards
        """
        plays = len(history)
        player_ind = plays % 2
        opponent_ind = 1 - player_ind

        # return payoff for terminal states
        utility = self.get_terminal_utility(player_ind, opponent_ind, plays, history, cards)
        
        if utility is not None:
            info_msg = 'History {}, terminal utility: {}'
            info_msg = info_msg.format(history, utility)
            # logger.info(info_msg)
            return utility

        info_set = '{}{}'.format(cards[player_ind], history)
        # Get information set node or create it if nonexistent
        node = self.get_info_set(info_set)

        # For each action, recursively call cfr with additional history and probability
        util, node_util = self.cfr_action(player_ind, node, history, p0, p1, cards)

        # For each action, compute and accumulate counterfactual regret
        self.accumulate_regret(player_ind, node, util, node_util, p0, p1)

        # info_msg = 'History: {} with utility: {:.2f}'
        # info_msg = info_msg.format(history, node_util)
        # logger.info(info_msg)

        return node_util

    def train(self, it):
        cards = np.asarray([i + 1 for i in range(3)])

        util = 0
        for i in range(it):
            if (i + 1) % 1000 == 0:
                info_msg = 'Running iteration {}'.format(i + 1)
                logger.info(info_msg)

            shuffle(cards)
            # this is Player 0's value of the game!
            util += self.cfr('', 1, 1, cards)

        print()
        info_msg = 'Average game value: {:.2f}'.format(util / it)
        logger.info(info_msg)
        
        info_msg = 'Information sets'
        logger.info(info_msg)

        for info_set, node in self.node_map.items():
            logger.info(node)


if __name__ == '__main__':
    it = 100000

    # cards = np.asarray([i + 1 for i in range(3)])
    # print('Unshuffled cards: \n{}'.format(cards))
    # shuffle(cards)
    # print('Shuffled cards: \n{}'.format(cards))

    instance = CFRInstance()
    instance.train(it)
