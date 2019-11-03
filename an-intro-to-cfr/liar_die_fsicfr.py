"""
Implementation of Fixed-Strategy-Iteration Counterfactual Regret Minimization for the modified Liar's dice game: Liar's die where only the immediately previous claim and the current claim are relevant to decision making.

This is an exercise from the paper An Introduction to Counterfactual Regret Minimization.
"""
import os
import numpy as np
import pandas as pd
from utils import make_logger
from tabulate import tabulate


logger = make_logger('liar_die_fsicfr.py')


DOUBT, ACCEPT = 0, 1
N_DIE_SIDES = 6
N_PLAYERS = 2
# doubt and accept
N_RESPONSE_ACTIONS = 2


def strategy2str(strategy, actions=None):
    str_ = ''
    for i in range(len(strategy)):
        if str_ != '':
            str_ += ', '
        action = actions[i] if actions is not None else i
        str_ += '{}: {:>6.2f}%'.format(action, strategy[i] * 100.)
    return str_


class Node:
    def __init__(self, n_actions, reach_player=0., reach_opponent=0.):
        self.n_actions = n_actions
        self.regret_sum = np.zeros(n_actions)
        self.strategy = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)

        # allows node to hold its utility value in order to allow backpropagation of
        # the utility to all predecessor nodes in the dynamic programming algorithm
        self.u = None
        # in the forward propogation of reach probabilities, the probabilities are 
        # accumulated in reach_player and reach_opponent
        # Note that these are a sum of probabilities rather than probabilities
        self.reach_player = reach_player
        self.reach_opponent = reach_opponent

    def get_strategy(self):
        normalizing_sum = 0.

        # accumulate positive regret sum at strategy
        self.strategy[:] = 0.
        positive_regret_ind = self.regret_sum > 0
        self.strategy[positive_regret_ind] = self.regret_sum[positive_regret_ind]
        normalizing_sum += self.strategy[positive_regret_ind].sum()

        if normalizing_sum > 0:
            self.strategy[positive_regret_ind] /= normalizing_sum
        else:
            self.strategy[:] = 1. / self.n_actions

        self.strategy_sum += self.reach_player * self.strategy

        strategy_str = strategy2str(self.strategy)
        err_msg = 'Strategy {} sums to {:.2f}, normalizing sum: {:.2f}'
        err_msg = err_msg.format(strategy_str, self.strategy.sum(), normalizing_sum)
        assert np.isclose(self.strategy.sum(), 1.), err_msg

        return self.strategy

    def get_average_strategy(self):
        normalizing_sum = self.strategy_sum.sum()

        if normalizing_sum > 0.:
            self.strategy_sum /= normalizing_sum
        else:
            self.strategy_sum[:] = 1. / self.n_actions

        return self.strategy_sum


class LiarDieTrainer:
    def __init__(self, n_sides):
        self.n_sides = n_sides
        # response nodes are indexed by the claim before the current claim and the current claim
        # claims are assumed to range from 1 through sides so that claim number 0 is used when a
        # claim does not exist
        # For example, the response node for an initial claim of c would be at index [0][c]
        # response_nodes.shape == [n_sides + 1][n_sides + 1]
        self.response_nodes = []
        # claim nodes are indexed by the accepted prior claim and the current die roll the claim 
        # decision concerns. The claim number 0 is used when a claim does not exist.
        # For example, the claim node for the initial claim of a game given a roll of r would be 
        # at index [0][r].
        # there are only n_sides prior claims because if prior_claim is n_sides, there is nothing to 
        # do but to DOUBT
        # claim_nodes.shape == [n_sides + 1][n_sides + 1]
        self.claim_nodes = []

        # my_claim refers to the claim before the current claim so there are n_sides + 1 of them
        for prev_claim in range(self.n_sides + 1):
            response_nodes_i = [None for _ in range(prev_claim + 1)]
            # opp_claim refers to the current claim, so there are n_sides of them
            for cur_claim in range(prev_claim + 1, self.n_sides + 1):
                # there is only one possible action when there is no prior claim
                # and only one possible action (i.e., ACCEPT) when the opponent 
                # claims the highest rank equal to the number of sides (i.e., DOUBT)
                n_actions = 1 if cur_claim == 0 or cur_claim == self.n_sides else 2
                response_node = Node(n_actions)
                response_nodes_i.append(response_node)
            self.response_nodes.append(response_nodes_i)

        # game would have finished if last claim was max claim
        for cur_claim in range(self.n_sides):
            claim_nodes_i = [None]
            for roll in range(1, self.n_sides + 1):
                # the number of legal claims remaining are the number of sides
                # minus the previous opponent claim
                claim_node = Node(self.n_sides - cur_claim)
                claim_nodes_i.append(claim_node)
            self.claim_nodes.append(claim_nodes_i)

    def get_init_rolls(self):
        return np.random.randint(low=1, high=self.n_sides + 1, size=self.n_sides)

    def init_start_prob(self, roll_arr):
        self.claim_nodes[0][roll_arr[0]].reach_player = 1.
        self.claim_nodes[0][roll_arr[0]].reach_opponent = 1.

    def visit_response_nodes_fwd(self, opp_claim, roll_arr):
        """Visit response nodes in forward pass. Accumulate:
            - reach probabilities at claim nodes of possible responses after opponent's claim
        """
        if opp_claim > 0:
            # my_claim refers to "my claim" before that could have brought the game to 
            # the current opponent's claim
            for my_claim in range(0, opp_claim):
                node = self.response_nodes[my_claim][opp_claim]
                assert isinstance(node, Node)

                # the action probability of a prior claim (my_claim) and a current claim (opp_claim)
                action_prob = node.get_strategy()

                # if opp_claim is not the last claim (n_sides), then we can either accept or doubt
                if opp_claim < self.n_sides:
                    # get the predetermined next roll upon opp_claim
                    next_roll = roll_arr[opp_claim]

                    # next claim node upon having opp_claim as prior claim and having next_roll
                    next_node = self.claim_nodes[opp_claim][next_roll]
                    assert isinstance(next_node, Node)

                    # the next_roll can be reached if the current player accepts claim
                    next_node.reach_player += action_prob[ACCEPT] * node.reach_player
                    next_node.reach_opponent += node.reach_opponent

    def visit_claim_nodes_fwd(self, opp_claim, roll_arr):
        """Visit claim nodes in forward pass. Accumulate:
            - 
        """
        # if my_claim == n_sides, then there is no strategy needed since there is
        # only a single legal action of DUDO
        if opp_claim < self.n_sides:
            # player roll after opponent has claimed opp_claim
            next_roll = roll_arr[opp_claim]

            # corresponding claim node with prior claim (opp_claim) and current roll (next_roll)
            node = self.claim_nodes[opp_claim][next_roll]
            assert isinstance(node, Node)

            # this should be over all possible claims, hence claim_node
            action_prob = node.get_strategy()

            # player can only claim higher than opp_claim and less or equal to n_sides
            for my_claim in range(opp_claim + 1, self.n_sides + 1):
                next_claim_prob = action_prob[my_claim - opp_claim - 1]
                if next_claim_prob > 0.:
                    next_node = self.response_nodes[opp_claim][my_claim]
                    assert isinstance(next_node, Node)

                    # the opponent had no decision making to reach this response node
                    # so it's just the counterfactual probability of reaching this node
                    next_node.reach_player += node.reach_opponent
                    # the probability of reaching this response node for the opponent 
                    # corresponds to the probability of player reaching the claim node 
                    # and the action probability that would take the game to this current node
                    next_node.reach_opponent += next_claim_prob * node.reach_player

    def visit_claim_nodes_bwd(self, opp_claim, roll_arr, regret):
        """Backpropagates by visiting the parent response nodes of the corresponding claim node
        of opp_claim and player roll.
        """
        assert len(regret) == self.n_sides + 1

        # if opp_claim == n_sides, then there was only the legal action of DOUBT
        if opp_claim < self.n_sides:
            roll = roll_arr[opp_claim]
            # the claim node reached after a prior opponent claim (opp_claim) and 
            # the current player roll
            node = self.claim_nodes[opp_claim][roll]
            assert isinstance(node, Node)

            # legal actions are all next legal claims
            action_prob = node.strategy
            strategy_str = strategy2str(action_prob)
            err_msg = 'Action prob {} sums to {:.2f}'
            err_msg = err_msg.format(strategy_str, action_prob.sum())
            assert np.isclose(action_prob.sum(), 1.), err_msg

            # recompute the node utility, so need to reset it
            node.u = 0.

            # possible next claims that player could have taken after accepting the 
            # opponent's claim as prior claim
            for my_claim in range(opp_claim + 1, self.n_sides + 1):
                action_ind = my_claim - opp_claim - 1
                # this would be the opponent's response node after player makes a 
                # claim of my_claim
                next_node = self.response_nodes[opp_claim][my_claim]
                assert isinstance(next_node, Node)

                # need to "flip the sign" to change from opponent's perspective to player
                child_util = -next_node.u
                # FIRST HALF of regret of player not having taken this action
                regret[my_claim] = child_util

                # the claim node's utility accumulates with this action probability and 
                # its eventual utility
                node.u += action_prob[action_ind] * child_util

            # SECOND HALF of regret of player not having taken actions
            # since regret = counterfactual action utility - expected utility
            regret[opp_claim + 1:] -= node.u
            node.regret_sum += node.reach_opponent * regret[opp_claim + 1:]

            # this is to reset node parameter for next round of forward pass
            node.reach_player = 0
            node.reach_opponent = 0

    def visit_response_nodes_bwd(self, opp_claim, roll_arr, regret):
        """Backpropagate starting from response node corresponding to opp_claim
        """
        assert len(regret) == 2

        # need to have an opponent's claim to respond to 
        if opp_claim > 0:
            # possible previous player claims leading up to current opponent claim
            for my_claim in range(opp_claim):
                # response node reached by having prior claim (my_claim) and current 
                # claim (opp_claim)
                node = self.response_nodes[my_claim][opp_claim]
                assert isinstance(node, Node)

                # legal actions are DOUBT and ACCEPT, or just one of them depending on the scenario
                action_prob = node.strategy
                strategy_str = strategy2str(action_prob)
                err_msg = 'Action prob {} sums to {:.2f}'
                err_msg = err_msg.format(strategy_str, action_prob.sum())
                assert np.isclose(action_prob.sum(), 1.), err_msg

                node.u = 0.
                # the roll that opponent got after accepting my_claim to make opp_claim
                roll = roll_arr[my_claim]
                # the utility if player doubts opp_claim, which is 1 (win) if opp_claim is 
                # higher than his roll, otherwise is -1 (lose)
                doubt_util = 1 if opp_claim > roll else -1

                # info_msg = 'Utility of doubting opp_claim [{}] with actual roll [{}]: {}'
                # info_msg = info_msg.format(opp_claim, roll, doubt_util)
                # logger.info(info_msg)

                # FIRST PART of computing regret of action DOUBT
                regret[DOUBT] = doubt_util
                node.u += action_prob[DOUBT] * doubt_util

                # player can only accept opp_claim if there is another next legal claim
                if opp_claim < self.n_sides:
                    # player roll after accepting opp_claim
                    roll = roll_arr[opp_claim]
                    # corresponding claim node after accepting opp_claim and making roll
                    next_node = self.claim_nodes[opp_claim][roll]
                    assert isinstance(next_node, Node)

                    # regret of accepting opp_claim and trying to "up" opponent
                    # FIRST PART of regret: counterfactual action utility
                    regret[ACCEPT] = next_node.u
                    node.u += action_prob[ACCEPT] * next_node.u

                    # SECOND PART of regret: expected utility of current node
                    regret[ACCEPT] -= node.u
                    # accumulate regret sum taking into counterfactual reach probability
                    node.regret_sum[ACCEPT] += node.reach_opponent * regret[ACCEPT]

                    # info_msg = 'ACCEPT regret sum: node utility: {:.2f}, reach_opp: {:.2f}, regret: {:.2f}'
                    # info_msg = info_msg.format(node.u, node.reach_opponent, regret[ACCEPT])
                    # logger.info(info_msg)


                # SECOND PART of computing the regret of action DOUBT
                regret[DOUBT] -= node.u
                # doesnt matter whether if node has 1 or 2 legal actions, DOUBT is 
                # always at index 0!
                node.regret_sum[DOUBT] += node.reach_opponent * regret[DOUBT]

                # resetting node reach probabilities for next iteration of forward pass
                node.reach_player = 0
                node.reach_opponent = 0

    def backpropagate(self, roll_arr, regret_claim, regret_response):
        for opp_claim in range(self.n_sides, -1, -1):
            # visit claim nodes backward
            self.visit_claim_nodes_bwd(opp_claim, roll_arr, regret_claim)

            # visit response nodes backward
            self.visit_response_nodes_bwd(opp_claim, roll_arr, regret_response)

    def accum_realization_weights_fwd(self, roll_arr, regret):
        for opp_claim in range(self.n_sides + 1):
            # visit response nodes forward
            self.visit_response_nodes_fwd(opp_claim, roll_arr)

            # visit claim nodes forward
            self.visit_claim_nodes_fwd(opp_claim, roll_arr)

    def train(self, n_iter):
        regret_claim = np.zeros(self.n_sides + 1)
        regret_response = np.zeros(2)
        roll_after_accepting_claim = None
        claim_value = 0.
        for it in range(n_iter):
            if it % 2000 == 0:
                info_msg = 'Doing iteration {}'.format(it)
                logger.info(it)
                self.print_resulting_strategy()

            # initialize rolls and starting probabilities
            roll_after_accepting_claim = self.get_init_rolls()
            self.init_start_prob(roll_after_accepting_claim)

            # accumulate realization weights forward
            self.accum_realization_weights_fwd(roll_after_accepting_claim, regret_claim)

            # backpropagate utilities, adjusting regrets and strategies
            self.backpropagate(roll_after_accepting_claim, regret_claim, regret_response)

            # reset strategy sums after half of training
            self.reset_strategy_sum(it, n_iter)

            claim_value_i = self.claim_nodes[0][roll_after_accepting_claim[0]].u
            claim_value += claim_value_i

            if (it + 1) % 1000 == 0:
                aux = claim_value / (it + 1)
                info_msg = 'Iteration {}: Avg. initial claim value: {:.5f}'
                info_msg = info_msg.format(it + 1, aux)
                logger.info(info_msg)

        # print resulting strategy
        self.print_resulting_strategy()

    def print_resulting_strategy(self):
        for init_roll in range(1, self.n_sides + 1):
            info_msg = 'Initial claim policy with roll {}: '.format(init_roll)
            node = self.claim_nodes[0][init_roll]
            avg_strategy = node.get_average_strategy()
            actions = [ i + 1 for i in range(self.n_sides) ]
            strategy_str = strategy2str(avg_strategy, actions=actions)

            info_msg += strategy_str

            print(info_msg)

        header = ['Old Claim', 'New Claim', 'Action Probabilities']
        table = list()
        for prev_claim in range(self.n_sides + 1):
            for cur_claim in range(prev_claim + 1, self.n_sides + 1):
                node = self.response_nodes[prev_claim][cur_claim]
                avg_strategy = node.get_average_strategy()
                if cur_claim == 0:
                    actions = ['ACCEPT']
                elif cur_claim == self.n_sides:
                    actions = ['DOUBT']
                else:
                    actions = [ 'DOUBT', 'ACCEPT' ]
                strategy_str = strategy2str(avg_strategy, actions)
                row = [prev_claim, cur_claim, strategy_str]
                table.append(row)

        table_str = tabulate(table, headers=header)
        print(table_str, end='\n\n')

        header = ['Old Claim', 'Roll', 'Action Probabilities']
        table = list()
        for cur_claim in range(self.n_sides):
            for roll in range(1, self.n_sides + 1):
                node = self.claim_nodes[cur_claim][roll]
                avg_strategy = node.get_average_strategy()
                actions = [ i + 1 for i in range(cur_claim, self.n_sides) ]
                strategy_str = strategy2str(avg_strategy, actions)

                row = [cur_claim, roll, strategy_str]
                table.append(row)

        table_str = tabulate(table, headers=header)
        print(table_str)

    def strategy2df(self):
        """Things recorded about two types of strategies.
        For response nodes:
        - old claim
        - new claim
        - action probabilities to accept or reject (-1. if not available)

        For claim nodes:
        - old claim
        - roll 
        - action probabilities for the different possible claims
        """
        header = [ 'old_claim', 'new_claim', 'doubt', 'accept' ] 
        rows = []
        for prev_claim in range(self.n_sides + 1):
            for cur_claim in range(prev_claim + 1, self.n_sides + 1):
                # there is only one possible action when there is no prior claim
                # and only one possible action (i.e., ACCEPT) when the opponent 
                # claims the highest rank equal to the number of sides (i.e., DOUBT)
                node = self.response_nodes[prev_claim][cur_claim]
                assert isinstance(node, Node)

                row = [prev_claim, cur_claim]
                if cur_claim == self.n_sides:
                    row.append(-1.)
                    row = np.concatenate((row, node.strategy))
                else:
                    row = np.concatenate((row, node.strategy))
                rows.append(row)
        response_df = pd.DataFrame.from_records(rows, columns=header)
        response_df[['old_claim', 'new_claim']] = response_df[['old_claim', 'new_claim']].astype(int)

        header = ['old_claim', 'roll',] + [str(i) for i in range(self.n_sides + 1)]
        rows = []
        for old_claim in range(self.n_sides):
            for roll in range(1, self.n_sides + 1):
                # the number of legal claims remaining are the number of sides
                # minus the previous opponent claim
                node = self.claim_nodes[old_claim][roll]
                assert isinstance(node, Node)
                row = [old_claim, roll] + [-1. for i in range(old_claim + 1)]
                row = np.concatenate((row, node.strategy))
                rows.append(row)
        claim_df = pd.DataFrame.from_records(rows, columns=header)
        claim_df[['old_claim', 'roll']] = claim_df[['old_claim', 'roll']].astype(int)

        return claim_df, response_df

    def reset_strategy_sum(self, it, n_iter):
        if it == n_iter / 2:
            for node_arr in self.response_nodes:
                for node in node_arr:
                    if node is not None:
                        node.strategy_sum[:] = 0
            
            for node_arr in self.claim_nodes:
                for node in node_arr:
                    if node is not None:
                        node.strategy_sum[:] = 0


if __name__ == '__main__':
    n_iter = 1000
    trainer = LiarDieTrainer(N_DIE_SIDES)

    trainer.train(n_iter)

    claim_df, response_df = trainer.strategy2df()

    outdir = os.path.join('results', 'liar-die-fsicfr')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    claim_fname = 'liar-die-fsicfr-n_die_sides_{}-it_{}-claim.csv'
    claim_fname = claim_fname.format(trainer.n_sides, n_iter)
    claim_fp = os.path.join(outdir, claim_fname)
    claim_df.to_csv(claim_fp, index=None, float_format='%.5f')

    response_fname = 'liar-die-fsicfr-n_die_sides_{}-it_{}-response.csv'
    response_fname = response_fname.format(trainer.n_sides, n_iter)
    response_fp = os.path.join(outdir, response_fname)
    response_df.to_csv(response_fp, index=None, float_format='%.5f')
