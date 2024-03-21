import os
from Axes.projection_functions import *
from Axes.bootstraping import *
import pandas as pd 
from matplotlib import pyplot as plt

def format_year(integer) :
    if len(str(integer)) > 4 :
        return int(str(integer)[:-2]+str(integer)[-1:])
    else :
        return integer

def plot_from_sources(df_all_grouped, axis, sources) :

    # Initialize dictionaries to store data
    cos = {source: [] for source in sources}
    CI_low = {source: [] for source in sources}
    CI_high = {source: [] for source in sources}

    # Loop through sources to gather data
    for source in sources:
        cos[source] = np.array(df_all_grouped[df_all_grouped["source"] == source]["cos axe " + str(axis)])
        CI_low[source] = np.array(df_all_grouped[df_all_grouped["source"] == source]["CI_" + str(axis) + "_inf"], dtype=float)
        CI_high[source] = np.array(df_all_grouped[df_all_grouped["source"] == source]["CI_" + str(axis) + "_sup"], dtype=float)

    # Plot
    plt.figure(figsize=(12, 6))

    for source in sources:
        x = df_all_grouped[df_all_grouped["source"] == source]["year"]
        plt.plot(x, cos[source],
                label="Cosine similarity on axis " + str(axis) + ' of ' + str(source), linewidth=2)
        plt.fill_between(x, CI_low[source], CI_high[source],
                        alpha=0.2)

    # Customize plot
    plt.title("On axis "+str(axis), fontsize=16)
    plt.xlabel("Année", fontsize=14)
    plt.ylabel("Similarité cosinus", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()  # Adjust layout
    plt.show()


def choose_projection_cos(axis=1, sources=['par','GUA', 'TE', 'DM', 'DE', 'MET'], focus_on_companies=None, curves_by_company=None) :
    
    df = pd.read_csv('data/current_dataframes/df')
    df_BT = pd.read_csv('data/current_dataframes/df_BT')

    # Validation check for source parameter
    if not sources == None :
        for source in sources :
            if not source in ['par', 'Con', 'Lab', 'GUA', 'TE', 'DM', 'DE', 'MET']:
                raise ValueError("source parameter must be one of ['par', 'Con', 'Lab', 'GUA', 'TE', 'DM', 'DE', 'MET']")
        
    if not focus_on_companies == None :

        for company in focus_on_companies :
            if not company in ['am', 'fb', 'go', 'ap', 'mi']:
                raise ValueError("company parameter must be one of ['am', 'fb', 'go', 'ap', 'mi']")

        df = df_BT[df_BT['class'].isin(focus_on_companies)]

    df_cbc = None
    if not curves_by_company == None :

        for company in curves_by_company :
            if not company in ['am', 'fb', 'go', 'ap', 'mi']:
                raise ValueError("company parameter must be one of ['am', 'fb', 'go', 'ap', 'mi']")

        df_inter = df_BT[df_BT['class'].isin(curves_by_company)]
        df_inter['source'] = df_inter['class']
        df_cbc = pd.concat((df_inter, df))

    df_inter = df[df['source'] == 'par']
    df_inter['source'] = df_inter['party']
    df = pd.concat((df, df_inter, df_cbc))

    df_all_grouped = df[["source", "year", "cos axe 1", "cos axe 2"]]

    if not sources == None and not curves_by_company ==  None:
        sources += curves_by_company
    if sources == None and not curves_by_company == None :
        sources = curves_by_company

    df_all_grouped = df_all_grouped[df_all_grouped["source"].isin(sources)]

    if 'par' in sources or 'Con' in sources or 'Lab' in sources :
        df_all_grouped = df_all_grouped[df_all_grouped['year'] < 2020]
    else :
        df_all_grouped['year'] = df_all_grouped['year'].apply(format_year)

    df_all_grouped = df_all_grouped.groupby(["source", "year"]).mean()
    df_all_grouped = df_all_grouped.reset_index()

    df_all_grouped = bootstrap(df_all_grouped, df, source_column="source", axis=axis)

    plot_from_sources(df_all_grouped, axis, sources)
