import numpy as np


ROCK, PAPER, SCISSOR = 0, 1, 2
NB_ACTIONS = 3
NB_ITERATIONS = 10000000


def get_strategy(strategy, regret_sum, strategy_sum):
    normalizing_sum = 0
    for a in range(NB_ACTIONS):
        strategy[a] = regret_sum[a] if regret_sum[a] > 0 else 0 
        normalizing_sum += strategy[a]

    for a in range(NB_ACTIONS):
        if normalizing_sum > 0:
            strategy[a] /= normalizing_sum
        else:
            strategy[a] /= NB_ACTIONS
        strategy_sum[a] += strategy[a]

    return strategy


def get_action(strategy):
    r = np.random.rand()
    a = 0
    cum_prob = 0
    while a < NB_ACTIONS - 1:
        cum_prob += strategy[a]
        if r < cum_prob:
            break
        a += 1
    return a


def get_avg_strategy(strategy_sum):
    avg_strategy = np.zeros(NB_ACTIONS)
    normalizing_sum = 0
    for a in range(NB_ACTIONS):
        normalizing_sum += strategy_sum[a]
    for a in range(NB_ACTIONS):
        if normalizing_sum > 0:
            avg_strategy[a] = strategy_sum[a] / normalizing_sum
        else:
            avg_strategy[a] = 1.0 / NB_ACTIONS
    return avg_strategy


def action2str(action):
    if action == ROCK:
        return 'ROCK'
    elif action == PAPER:
        return 'PAPER'
    else:
        return 'SCISSOR'


def get_winner(my_action, opp_action):
    if my_action == opp_action:
        return None
    elif my_action > opp_action:
        return 'Left'
    else:
        return 'Right'


def strategy2str(strategy):
    s = 'Rock: {:.2f}%, Paper: {:.2f}%, Scissor: {:.2f}%'
    s = s.format(strategy[0] * 100., strategy[1] * 100., strategy[2] * 100.)
    return s


def update_action_utility(action_utility, action0, action1):
    action_utility[action1] = 0
    is_scissor = action1 == SCISSOR
    action_utility[ROCK if is_scissor else action1 + 1] = 1
    is_rock = action1 == ROCK
    action_utility[SCISSOR if is_rock else action1 - 1] = -1


if __name__ == '__main__':
    my_strategy = np.zeros(NB_ACTIONS)
    my_strategy_sum = np.zeros(NB_ACTIONS)
    my_regret_sum = np.zeros(NB_ACTIONS)
    my_action_utility = np.zeros(NB_ACTIONS)

    opp_strategy = np.zeros(NB_ACTIONS)
    opp_strategy_sum = np.zeros(NB_ACTIONS)
    opp_regret_sum = np.zeros(NB_ACTIONS)
    opp_action_utility = np.zeros(NB_ACTIONS)

    win_stats = np.zeros(2)

    for it in range(NB_ITERATIONS):
        # get regret matched mixed strategy actions
        my_strategy = get_strategy(my_strategy, my_regret_sum, my_strategy_sum)
        opp_strategy = get_strategy(opp_strategy, opp_regret_sum, opp_strategy_sum)

        my_action = get_action(my_strategy)
        opp_action = get_action(opp_strategy)

        # compute action utilities
#         my_action_utility[opp_action] = 0
#         is_scissor = opp_action == SCISSOR
#         my_action_utility[ROCK if is_scissor else opp_action + 1] = 1
#         is_rock = opp_action == ROCK
#         my_action_utility[SCISSOR if is_rock else opp_action - 1] = -1
        update_action_utility(my_action_utility, my_action, opp_action)
        update_action_utility(opp_action_utility, opp_action, my_action)

        info_msg = 'Iteration {}: {} won: {} (left) vs. {} (right)'
        winner = get_winner(my_action, opp_action)
        info_msg = info_msg.format(it, winner, action2str(my_action), action2str(opp_action))
        print(info_msg)
        if winner == 'Left':
            win_stats[0] += 1
        elif winner == 'Right':
            win_stats[1] += 1
        else:
            win_stats += 0.5

        # accumulate action regrets
        for a in range(NB_ACTIONS):
            my_regret_sum[a] += my_action_utility[a] - my_action_utility[my_action]

        for a in range(NB_ACTIONS):
            opp_regret_sum[a] += opp_action_utility[a] - opp_action_utility[opp_action]

    my_avg_strategy = get_avg_strategy(my_strategy_sum)
    info_msg = 'My average strategy: \n{}'.format(strategy2str(my_avg_strategy))
    print(info_msg)
    opp_avg_strategy = get_avg_strategy(opp_strategy_sum)
    info_msg = 'Opp average strategy: \n{}'.format(strategy2str(opp_avg_strategy))
    print(info_msg)

    print('Win stats: {}, win stats perc: {}'.format(win_stats, win_stats / win_stats.sum()))
