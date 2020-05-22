import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.evaluation.general import latexify

if __name__ == '__main__':
    df = pd.read_csv('corpus.metadata.modified.csv')
    # tokens per city, tokens per topic (deeds,law), tokens per origin, tokens per time
    outdir = '../../../../Thesis/eval/'

    latexify(columns=2)

    ax = sns.barplot(x='City', y='Tokencount',
                     data=df.groupby('City').sum().sort_values('Tokencount', ascending=False).reset_index())
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode='anchor')
    ax.set_ylabel('')
    plt.tight_layout()
    ax.get_figure().savefig(outdir + 'City.pdf')
    plt.clf()

    ax = sns.barplot(x='Type', y='Tokencount', data=df.groupby('Type').sum().reset_index())
    ax.set_ylabel('')
    plt.tight_layout()
    ax.get_figure().savefig(outdir + 'Type.pdf', bbox_layout='tight')
    plt.clf()

    ax = sns.barplot(x='Year', y='Tokencount', data=df.groupby('Year').sum().reset_index(), palette='Greens')
    # ax = sns.lineplot(x='Year', y='Tokencount', data=df.groupby('Year').sum().reset_index())
    ax.set_ylabel('')
    plt.tight_layout()
    ax.get_figure().savefig(outdir + 'Year.pdf', bbox_layout='tight')

    with open(outdir + 'metadata.tex', 'w') as f:
        df = df['sigle,name,Year,Tokencount'.split(',')]
        df.to_latex(buf=f, escape=False, multicolumn_format='c', multirow=True, index=False)
