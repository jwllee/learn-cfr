import numpy as np
import time


from utils import make_logger


logger = make_logger('liar_dices_cfr.py')


#================================================================================
# Utility functions
#================================================================================

def infoset2int(player_roll, is_claimed, cur_total_n_dices):
    """Convert information set to hash number.
    :param player_roll int: represent the rolls of dices as an integer, assuming that the rolls were sorted in ascending order by their ranks.
    :param is_claimed array_like: boolean array of all claims that have been done so far in the current round.
    :param cur_total_n_dices int: Current total number of dices.
    """
    infoset_num = player_roll

    n_actions = len(is_claimed)
    for action in range(n_actions - 2, -1, -1):
        infoset_num = 2 * infoset_num + int(is_claimed[action])
    infoset_num = 2 * infoset_num + int(cur_total_n_dices)

    return infoset_num


def infoset2str(player_roll, is_claimed, cur_total_n_dices, prev_claim):
    """Convert information set to hash string.
    :param player_roll int: represent the rolls of dices as an integer, assuming that the rolls were sorted in ascending order by their ranks.
    :param is_claimed array_like: boolean array of all claims that have been done so far in the current round.
    :param cur_total_n_dices int: Current total number of dices.
    """
    infoset_str = str(player_roll) + '|'

    n_actions = len(is_claimed)
    for action in range(n_actions):
        infoset_str += str(int(is_claimed[action]))
    infoset_str += '|' + str(int(cur_total_n_dices))
    # infoset_str += '|' + str(int(prev_claim))

    return infoset_str


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


#================================================================================


class GameParameter:
    def __init__(self, n_players, n_dices, n_dice_sides, n_rounds=np.inf, wild_rank=6):
        """
        :param n_players int: Number of players
        :param n_dices int: Number of dices per player
        :param n_dice_sides int: Number of sides that each die has
        :param n_rounds int: Max number of rounds
        :param wild_rank int: Die rank that is a wild card
        """
        self.n_players = n_players
        self.n_dices = n_dices
        self.n_dice_sides = n_dice_sides
        self.n_rounds = n_rounds
        self.wild_rank = wild_rank

    @property
    def total_n_dices(self):
        return self.n_players * self.n_dices

    @property
    def n_actions(self):
        return self.n_players * self.n_dices * self.n_dice_sides + 1

    def get_player_roll_template(self):
        temp = []
        for player in range(self.n_players):
            temp.append([10 ** i for i in range(self.n_dices)])
        temp = np.asarray(temp)
        return temp

    def get_claim_rank(self):
        claim_rank = []
        for _ in range(self.n_players * self.n_dices):
            claim_rank += [i + 1 for i in range(self.n_dice_sides)]
        claim_rank = np.asarray(claim_rank)
        return claim_rank

    def get_claim_num(self):
        claim_num = []
        for i in range(self.n_players * self.n_dices):
            claim_num += [i + 1 for _ in range(self.n_dice_sides)]
        claim_num = np.asarray(claim_num)
        return claim_num

    def get_dices(self):
        low, high = 1, self.n_dice_sides + 1
        shape = (self.n_players, self.n_dices)
        dices = np.random.randint(low, high, shape)
        # sort ascendingly per row
        dices = np.sort(dices, axis=0)
        return dices

    def get_action_str(self):
        return None

    def __repr__(self):
        repr_ = '{}(n_players: {}, n_dices: {}, n_dice_sides: {}, n_rounds: {})'
        repr_ = repr_.format(GameParameter.__name__,
                             self.n_players,
                             self.n_dices,
                             self.n_dice_sides,
                             self.n_rounds)
        return repr_


class InfoSetContainer:
    def __init__(self, infoset_id, player_roll, is_claimed, 
                 cur_total_n_dices, game, prev_claim=-1):
        self.game = game
        self.player_roll = player_roll.copy()
        self.is_claimed = is_claimed.copy()
        self.cur_total_n_dices = cur_total_n_dices
        self.prev_claim = prev_claim
        self.infoset_id = infoset_id

        n_actions = len(is_claimed)
        self.regret_sum = np.zeros(n_actions)
        self.strategy = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)

        # make valid claim indices
        n_actions = len(self.is_claimed)
        if not self.is_claimed.any():
            self.legal_actions = list(range(n_actions - 1))
        else:
            self.legal_actions = list()
            last_claim = np.argwhere(self.is_claimed).ravel()[-1]
            for action in range(last_claim + 1, n_actions - 1):
                if self.game.claim_num[action] > self.cur_total_n_dices:
                    break
                self.legal_actions.append(action)
            self.legal_actions.append(n_actions - 1)

    def get_strategy(self, reach_prob):
        """Get strategy for information set.
        :param reach_prob float: Probability of reaching the information set, this can be different at different times since there are multiple histories within an information set.
        """
        normalizing_sum = 0.

        n_actions = len(self.is_claimed)

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
            # need to incorporate the reach probability to this particular
            # history in the accumulation for the overall strategy
            
            self.strategy_sum[action] += reach_prob * self.strategy[action]

        return self.strategy

    def get_average_strategy(self):
        n_actions = len(self.is_claimed)
        avg_strategy = np.zeros(n_actions)

        normalizing_sum = 0.
        for action in self.legal_actions:
            normalizing_sum += self.strategy_sum[action]

        for action in self.legal_actions:
            if normalizing_sum > 0:
                avg_strategy[action] = self.strategy_sum[action] / normalizing_sum
            else:
                avg_strategy[action] = 1. / len(self.legal_actions)

        return avg_strategy

    def __repr__(self):
        repr_ = '{}({}): {}'
        avg_strategy = self.get_average_strategy()
        avg_strategy_str = '['
        n_actions = len(self.is_claimed)
        for action in range(n_actions):
            if avg_strategy_str != '[':
                avg_strategy_str += ', '

            str_i = '{}: {:.2f}%'.format(
                action, avg_strategy[action] * 100.
            )
            avg_strategy_str += str_i
        avg_strategy_str += ']'
        repr_ = repr_.format(self.infoset_id,
                             self.player_roll, 
                             avg_strategy_str)
        return repr_

    def __str__(self):
        return str(repr(self))


def strategy2str(strategy):
    str_ = '['
    for action in range(len(strategy)):
        if str_ != '[':
            str_ += ', '
        str_i = '{}: {:.2f}%'.format(
            action, strategy[action] * 100.
        )
        str_ += str_i
    str_ += ']'
    return str_


class CFRInstance:
    def __init__(self, game_params):
        assert isinstance(game_params, GameParameter)

        self.game_params = game_params
        self.claim_rank = game_params.get_claim_rank()
        self.claim_num = game_params.get_claim_num()
        self.rolls_template = game_params.get_player_roll_template()

        self.infoset_map = dict()

    def set_player_roll(self, dices):
        # update roll
        player_roll = dices * self.rolls_template
        player_roll = player_roll.sum(axis=1)

        player_roll_shape = player_roll.shape
        n_players_shape = (self.game_params.n_players, )
        err_msg = 'Player rolls shape {} != {} n_players'
        err_msg = err_msg.format(player_roll_shape, n_players_shape)
        assert player_roll_shape == n_players_shape, err_msg

        return player_roll

    def roll_dices(self, dices):
        mask = dices > 0
        # places where there are zero dices will remain with zero dices
        dices[:] = mask * self.game_params.get_dices()

    def get_terminal_utility(self, claimer_history, history, player_n_dices, dices):
        # info_msg = 'Get terminal utility, Round claimer history: {}, Player_n_dices: {}'
        # info_msg = info_msg.format(claimer_history[-1], player_n_dices)
        # logger.info(info_msg)

        round_no = len(history)
        cur_claimer_history = claimer_history[-1]
        player = cur_claimer_history[-1]

        # check the number of remaining dices of each player
        n_active = (player_n_dices > 0).sum()
        # one player with dices
        if n_active == 1:
            if player_n_dices[player] > 0:
                # so that it is zero-sum
                return self.game_params.n_players - 1
            else:
                return -1 

        # might be hitting the max iterations
        if round_no >= self.game_params.n_rounds:
            # player with max number of dices wins
            max_n_dices = max(player_n_dices)
            n_max_players = sum(player_n_dices == max_n_dices)

            if player_n_dices[player] == max_n_dices:
                return (self.game_params.n_players - 1)/ n_max_players
            else:
                # divide the loss over all losing players
                return (self.game_params.n_players - 1) / (self.game_params.n_players - n_max_players)

    def get_infoset(self, infoset_id, player_roll, 
                    is_claimed, cur_total_n_dices, prev_claim):

        node = self.infoset_map.get(infoset_id)
        if node is None:
            node = InfoSetContainer(
                infoset_id, player_roll, is_claimed,
                cur_total_n_dices, self, prev_claim, 
            )
            self.infoset_map[infoset_id] = node
        return node

    def get_next_player(self, claimer_history, player_rolls):
        if len(claimer_history) == 1:
            return claimer_history[0]

        last_player = claimer_history[-1]
        next_player = last_player
        candidate = next_player
        n_checked = 0
        while next_player == last_player:
            candidate = (candidate + 1) % self.game_params.n_players
            if player_rolls[candidate] > 0: # has at least one die
                next_player = candidate

            n_checked += 1
            if n_checked >= self.game_params.n_players:
                err_msg = 'No other active player than {}: {}'
                err_msg = err_msg.format(last_player, player_rolls)
                raise ValueError(err_msg)

        if next_player == last_player:
            err_msg = 'No other active player than {}: {}'
            err_msg = err_msg.format(last_player, player_rolls)
            raise ValueError(err_msg)

        assert isinstance(last_player, int), '{} is not int'.format(last_player)
        assert isinstance(next_player, int), '{} is not int'.format(next_player)

        return next_player

    def cfr_action(self, node, 
                   claimer_history, reach_probs, player_rolls,
                   dices, player_n_dices, history):
        """For each action, recursively call cfr with additional history and probability.

        """
        cur_claimer_history = claimer_history[-1]
        player = self.get_next_player(cur_claimer_history, player_rolls)
        reach_prob = reach_probs[player]
        cur_total_n_dices = player_n_dices.sum()

        strategy = node.get_strategy(reach_prob)
        util = np.zeros(self.game_params.n_actions)
        node_util = 0.
        is_claimed = history[-1]

        # info_msg = 'Strategy for player {} at round with claim history: {} \nclaimed: {} \nstrategy: {}'
        # strategy_str = strategy2str(strategy)
        # info_msg = info_msg.format(player, cur_claimer_history, is_claimed, strategy_str)
        # logger.info(info_msg)
        # time.sleep(10) 

        for action in node.legal_actions:
            player = self.get_next_player(cur_claimer_history, player_rolls)

            history_copied = history.copy()
            claimer_history_copied = claimer_history.copy()
            reach_probs_copied = reach_probs.copy()
            player_n_dices_copied = player_n_dices.copy()
            player_rolls_copied = player_rolls.copy()
            dices_copied = dices.copy()

            history_copied[-1][action] = True
            claimer_history_copied[-1].append(player)
            reach_probs_copied[player] *= strategy[action]

            util[action] = -self.cfr(claimer_history_copied, reach_probs_copied, 
                                     player_rolls_copied, dices_copied, player_n_dices_copied, history_copied)

            node_util += strategy[action] * util[action]

        return util, node_util

    def accumulate_regret(self, player, node, util, node_util, reach_prob):
        """For each action, compute and accumulate counterfactual regret.

        :param node InfoSetContainer: information set container
        :param util array_like: counterfactual utility of actions
        :param node_util float: counterfactual utility of information set container
        :param reach_prob array_like: reach probability for each player
        """
        err_msg = '{!r} is not instance of {}'.format(node, InfoSetContainer.__name__)
        assert isinstance(node, InfoSetContainer), err_msg

        # info_msg = 'Infoset: {}, Reach probability: {}'
        # reach_prob_str = strategy2str(reach_prob)
        # info_msg = info_msg.format(node.infoset_id, reach_prob_str)
        # logger.info(info_msg)

        # time.sleep(5)

        for action in range(self.game_params.n_actions):
            regret = util[action] - node_util
            player_prob = reach_prob[player]
            reach_prob[player] = 1.
            counterfactual_reach_prob = reach_prob.prod()
            reach_prob[player] = player_prob
            node.regret_sum[action] += counterfactual_reach_prob * regret

    def get_num_of_rank(self, dices, claimed_rank):
        raveled = np.ravel(dices)
        to_count = raveled[(raveled == claimed_rank) | (raveled == self.game_params.wild_rank)]
        return len(to_count)

    def deactivate_player_dices(self, player, n, dices, player_n_dices):
        player_n_dices[player] -= n
        if player_n_dices[player] < 0:
            player_n_dices[player] = 0

        # can throw exception but we assume that player has dices to deactivate
        start_ind = int(np.argwhere(dices[player] > 0).ravel()[0])
        if start_ind + n <= self.game_params.n_dices:
            end_ind = int(start_ind + n)
        else:
            end_ind = int(self.game_params.n_dices)
        dices[player, start_ind:end_ind] = 0

    def handle_round_end(self, claimer_history, dices, player_n_dices, history):
        is_claimed = history[-1]

        if not is_claimed[-1]:
            return 

        # check that there have been at least 2 claiming players
        cur_claimer_history = claimer_history[-1]
        err_msg = 'Less than two claiming players'
        assert len(cur_claimer_history) >= 2, err_msg
        
        # Check for the three possible conditions
        # 1. Actual rank count exceeds the challenged claim:
        # Challenger loses a number of dices equal to the difference between the 
        # actual rank count and the claim count.
        # 
        # 2. Actual rank count is less than the challenged claim
        # The challenged player loses a number of dices equal to the difference 
        # between the claim count and the actual rank count.
        #
        # 3. Actual rank count is equal to the challenged claim
        # Every player except the challenged player loses a single die. 
        
        challenger = cur_claimer_history[-1]
        claimer = cur_claimer_history[-2]

        last_claim = np.argwhere(is_claimed).ravel()[-2]
        claimed_rank = self.claim_rank[last_claim]
        claimed_num = self.claim_num[last_claim]
        actual_num = self.get_num_of_rank(dices, claimed_rank)
        diff = abs(actual_num - claimed_num)

        # Condition 1:
        if actual_num > claimed_num:
            winner = claimer
            self.deactivate_player_dices(challenger, diff, dices, player_n_dices)
        # Condition 2:
        elif actual_num < claimed_num:
            winner = challenger
            self.deactivate_player_dices(claimer, diff, dices, player_n_dices)
        # Condition 3:
        else:
            winner = claimer
            for player in range(self.game_params.n_players):
                if player == claimer:
                    continue
                self.deactivate_player_dices(player, 1, dices, player_n_dices)

        round_no = len(history)
        # info_msg = 'Player {} won round {}, Claim history: {}, Player_n_dices: {}'
        # info_msg = info_msg.format(winner, round_no, history[-1], player_n_dices)
        # logger.info(info_msg)

        # next round
        # info_msg = 'Starting round {}'.format(len(history) + 1)
        # logger.info(info_msg)

        is_claimed = np.zeros(self.game_params.n_actions, dtype=np.bool)
        history.append(is_claimed)
        claimer_history.append([winner])
        # roll dices
        self.roll_dices(dices)

    def cfr(self, claimer_history, reach_probs, player_rolls, dices, player_n_dices, history):
        # info_msg = 'Current round history: {} \nCurrent claimer history: {}'
        # info_msg = info_msg.format(history[-1], claimer_history[-1])
        # logger.info(info_msg)

        # check for end of round with someone calling Dudo
        self.handle_round_end(claimer_history, dices, player_n_dices, history)

        round_no = len(history)
        is_claimed = history[-1]
        # return payoff for terminal states
        utility = self.get_terminal_utility(claimer_history, history, player_n_dices, dices)

        if utility is not None:
            # info_msg = 'History {}, terminal utility: {}'
            # info_msg = info_msg.format(history, utility)
            # logger.info(info_msg)
            return utility 

        cur_claimer_history = claimer_history[-1]
        cur_total_n_dices = player_n_dices.sum()
        player = self.get_next_player(cur_claimer_history, player_rolls)
        player_roll = player_rolls[player]
        if not is_claimed.any():
            prev_claim = -1
        else:
            prev_claim = np.argwhere(is_claimed).ravel()[-1]

        infoset_id = infoset2str(player_roll, is_claimed, cur_total_n_dices, prev_claim)
        # Get information set node or create it if nonexistent
        node = self.get_infoset(infoset_id, player_roll, is_claimed, cur_total_n_dices, prev_claim)

        # For each action, recursively call cfr with additional history and probability
        util, node_util = self.cfr_action(
            node, claimer_history, reach_probs, player_rolls, 
            dices, player_n_dices, history
        )

        # For each action, compute and accumulate counterfactual regret
        self.accumulate_regret(player, node, util, node_util, reach_probs)

        # info_msg = 'History: {} with utility: {:.2f}'
        # info_msg = info_msg.format(history, node_util)

        return node_util

    def train(self, it):
        util = 0
        for i in range(it):
            if (i + 1) % 100 == 0:
                info_msg = 'Running iteration {}, Current average game value: {}'
                info_msg = info_msg.format(i + 1, util / i)
                logger.info(info_msg)

            dices = self.game_params.get_dices()
            player_rolls = self.set_player_roll(dices)
            player = 0
            claim_players = [[player]]
            reach_probs = np.ones(self.game_params.n_players)
            player_n_dices = np.zeros(self.game_params.n_players) 
            player_n_dices += self.game_params.n_dices
            history = []
            is_claimed = np.zeros(self.game_params.n_actions, dtype=np.bool)
            history.append(is_claimed)

            # this is player's value of the game
            util_i = self.cfr(claim_players, reach_probs, player_rolls, 
                             dices, player_n_dices, history)
            util += util_i

        print()
        info_msg = 'Average game value: {:.4f}'.format(util / it)
        logger.info(info_msg)

        info_msg = 'Information sets'
        logger.info(info_msg)

        for info_set, node in self.infoset_map.items():
            logger.info(node)

    def __repr__(self):
        repr_ = '{}({})'
        repr_ = repr_.format(CFRInstance.__name__, 
                             repr(self.game_params))
        return repr_

    def describe(self):
        str_ = 'Description of {}:\n'.format(CFRInstance.__name__)
        str_ += 'repr: {}\n'.format(repr(self))
        print(str_)


if __name__ == '__main__':
    it = 1000

    n_players, n_dices, n_dice_sides = 2, 1, 6
    n_rounds = np.inf

    game_params = GameParameter(
        n_players,
        n_dices,
        n_dice_sides,
        n_rounds
    )
    instance = CFRInstance(game_params)
    instance.train(it)
