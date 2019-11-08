import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import os, matplotlib


matplotlib.use('Agg')


IT = 1000000
RESULTS_DIR = os.path.join('results', 'liar-die-fsicfr')
CLAIM_FNAME = 'liar-die-fsicfr-n_die_sides_6-it_{}-claim.csv'.format(IT)
RESPONSE_FNAME = 'liar-die-fsicfr-n_die_sides_6-it_{}-response.csv'.format(IT)
EXPLOITABILITY_FNAME = 'liar-die-fsicfr-n_die_sides_6-it_{}-exploitability.csv'.format(IT)

CLAIM_FP = os.path.join(RESULTS_DIR, CLAIM_FNAME)
RESPONSE_FP = os.path.join(RESULTS_DIR, RESPONSE_FNAME)
EXPLOITABILITY_FP = os.path.join(RESULTS_DIR, EXPLOITABILITY_FNAME)


IMG_DIR = os.path.join('img', 'liar-die-fsicfr')
if not os.path.isdir(IMG_DIR):
    os.makedirs(IMG_DIR)


if __name__ == '__main__':
    claim_df = pd.read_csv(CLAIM_FP)
    response_df = pd.read_csv(RESPONSE_FP)
    exploit_df = pd.read_csv(EXPLOITABILITY_FP)

    print(claim_df.head())
    print(response_df)
    print(exploit_df.head())

    # plot response results
    response_df.replace(to_replace=-1, value=0, inplace=True)
    response_df['metric'] = response_df['doubt'] - response_df['accept']
    print(response_df.head())

    response = response_df.pivot('old_claim', 'new_claim', 'metric')
    fig, ax = plt.subplots()
    cmap = sns.color_palette('coolwarm', 7)
    cax = sns.heatmap(response, ax=ax, cmap=cmap, linewidths=2, cbar=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top + 0.5)
    ax.invert_yaxis()

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Accept', '50/50', 'Doubt'])

    response_heatmap_fname = RESPONSE_FNAME + '-heatmap.svg'
    response_heatmap_fp = os.path.join(IMG_DIR, response_heatmap_fname)
    fig.savefig(response_heatmap_fp)

    # plot claim results
    stack_claim_df = claim_df.drop(columns='0').set_index(['old_claim', 'roll'])
    # stack to pivot roll columns as index
    stack_claim_df = stack_claim_df.stack()
    # reset player roll index so that it becomes a column
    stack_claim_df = stack_claim_df.reset_index(level=2)
    stack_claim_df.columns = ['action', 'probability']
    stack_claim_df = stack_claim_df.reset_index()
    stack_claim_df.replace(to_replace=-1, value=0, inplace=True)
    # print(stack_claim_df.head(10))

    fig, ax = plt.subplots()
    g = sns.FacetGrid(stack_claim_df, row='roll', col='old_claim', margin_titles=True)
    actions = stack_claim_df['action'].unique()
    g.map(sns.barplot, 'action', 'probability', order=actions)
    claim_facet_fname = CLAIM_FNAME + '-facetgrid.svg'
    claim_facet_fp = os.path.join(IMG_DIR, claim_facet_fname)
    g.savefig(claim_facet_fp)

    # plot exploitability results
    fig, ax = plt.subplots()
    exploit_df['iteration'] = exploit_df['iteration'] / 1000
    exploit_df.rename(columns={'iteration': 'iteration (thousands)'}, inplace=True)
    sns.regplot(x='iteration (thousands)', y='exploitability', data=exploit_df, ax=ax)
    exploit_fname = EXPLOITABILITY_FNAME + '-lmplot.svg'
    exploit_fp = os.path.join(IMG_DIR, exploit_fname)
    fig.savefig(exploit_fp)
