import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import os, matplotlib

matplotlib.use('Agg')


RESULTS_FNAME = 'n_players_2-n_dices_1-n_die_sides_6-it_360000.csv'
RESULTS_FP = os.path.join('results', RESULTS_FNAME)


IMG_DIR = 'img'
if not os.path.isdir(IMG_DIR):
    os.mkdir(IMG_DIR)


if __name__ == '__main__':
    df = pd.read_csv(RESULTS_FP)

    print(df.head())

    # plot the strategy and regret sum of first player
    is_claimed_cols = list(filter(lambda c: c.startswith('is_claimed'), df.columns))
    strategy_cols = list(filter(lambda c: c.startswith('avg_strategy'), df.columns))
    regret_cols = list(filter(lambda c: c.startswith('regret'), df.columns))

    is_first = df.loc[:, is_claimed_cols].apply(lambda r: not r.any(), axis=1)
    # print(is_first)

    first_df = df.loc[(is_first), :]
    # print(first_df.head())

    cols = ['player_roll', ] + strategy_cols
    first_strat_df = first_df[cols].sort_values('player_roll')
    first_strat_df = first_strat_df.set_index('player_roll')
    first_strat_stacked_df = first_strat_df.stack().reset_index(drop=False)
    # first_strat_stacked_df.drop(labels='level_0', axis=1, inplace=True)
    first_strat_stacked_df.columns = ['player_roll', 'claim', 'likelihood']
    remove_prefix = lambda s: s.replace('avg_strategy_', '')
    first_strat_stacked_df['claim'] = first_strat_stacked_df['claim'].apply(remove_prefix)
    print(first_strat_stacked_df.tail(20))

    fig, ax = plt.subplots(figsize=(10, 8))
    first_strat_stacked_df = first_strat_stacked_df.pivot('player_roll', 'claim', 'likelihood')
    cmap = sns.color_palette('coolwarm', 7)
    sns.heatmap(first_strat_stacked_df, ax=ax, cmap=cmap, linewidths=1.5)
    n = 6

    # Turn off spines and create white grid
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    ax.set_yticks(np.arange(n + 1.05)-.5, minor=True)
    strat_fname = 'first_player_strategy'
    strat_fp = os.path.join(IMG_DIR, '{}.svg'.format(strat_fname))
    i = 1
    while os.path.exists(strat_fp):
        strat_fp = os.path.join(IMG_DIR, '{}-{}.svg'.format(strat_fname, i))
        i += 1
    fig.savefig(strat_fp, bbox_inches='tight')

    # plot the regret sum
    cols = ['player_roll',] + regret_cols
    first_regret_df = first_df[cols].sort_values('player_roll')
    first_regret_df = first_regret_df.set_index('player_roll')
    first_regret_stacked_df = first_regret_df.stack().reset_index(drop=False)
    first_regret_stacked_df.columns = ['player_roll', 'claim', 'regret_sum']
    remove_prefix = lambda s: s.replace('regret_', '')
    first_regret_stacked_df['claim'] = first_regret_stacked_df['claim'].apply(remove_prefix)

    fig, ax = plt.subplots(figsize=(10, 8))
    first_regret_stacked_df = first_regret_stacked_df.pivot('player_roll', 'claim', 'regret_sum')
    cmap = sns.color_palette('coolwarm', 7)
    sns.heatmap(first_regret_stacked_df, ax=ax, cmap=cmap, linewidths=1.5)
    n = 6

    ax.set_yticks(np.arange(n + 1.05)-.5, minor=True)
    regret_fname = 'first_player_regret'
    regret_fp = os.path.join(IMG_DIR, '{}.svg'.format(regret_fname))
    i = 1
    while os.path.exists(regret_fp):
        regret_fp = os.path.join(IMG_DIR, '{}-{}.svg'.format(regret_fname, i))
        i += 1
    fig.savefig(regret_fp, bbox_inches='tight')

    strategy_fp = 'first_strategy.csv'
    first_strat_df.to_csv(strategy_fp)
    regret_fp = 'first_regret.csv'
    first_regret_df.to_csv(regret_fp)
