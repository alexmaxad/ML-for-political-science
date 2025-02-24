from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
from sklearn.metrics import silhouette_score
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def silhouette_score_(k_rng, data):
    """
    Calculate and plot the silhouette scores for a range of cluster sizes.

    This function iterates over a range of values for 'k' (number of clusters) and calculates
    the silhouette score for each. The silhouette score is a measure of how similar an object
    is to its own cluster compared to other clusters. The scores are then plotted to help identify
    the optimal number of clusters.

    Parameters:
    - k_rng (list): A list of integers representing the range of cluster numbers to evaluate.
    - data (array-like): The dataset to be clustered.

    The function outputs a plot of silhouette scores across different values of 'k'.
    """
    sil = []
    for k in k_rng:
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.predict(data)
        sil.append(silhouette_score(data, labels))

    plt.plot(k_rng, np.array(sil))
    plt.xlabel("k")
    plt.ylabel("Silhouette index")
    plt.show()
    plt.close()



def sse_scaler_(k_rng, data):
    """
    Calculate and plot the sum of squared errors (SSE) for a range of cluster sizes.

    This function iterates over a range of 'k' values, fitting a KMeans model for each and calculating
    the SSE. The SSE is a measure of how close each point in a cluster is to the centroid of that cluster,
    acting as a criterion for selecting the number of clusters. The SSE values are plotted against 'k'.

    Parameters:
    - k_rng (list): A list of integers specifying the range of cluster numbers to evaluate.
    - data (array-like): The dataset to be clustered.

    Outputs a plot of SSE values across different 'k' values.
    """
    sse_scaler = []
    for k in k_rng:
        km = KMeans(n_clusters=k).fit(data)
        sse_scaler.append(km.inertia_)

    plt.plot(k_rng, sse_scaler)
    plt.xlabel("k")
    plt.ylabel("Sum of squared error")
    plt.show()
    plt.close()



def plot_clusters_on_pc_spectral(number_of_clusters, data):
    """
    Perform Spectral Clustering and visualize the clusters in 2D using PCA.

    This function applies Spectral Clustering to the dataset and uses PCA to reduce
    the dimensionality of the data to two principal components for visualization purposes.
    The resulting clusters are then plotted in a 2D scatter plot.

    Parameters:
    - number_of_clusters (int): The number of clusters to generate.
    - data (array-like): The dataset to be clustered and visualized.
    """
    model = SpectralClustering(n_clusters=number_of_clusters, assign_labels="discretize",
                               random_state=0, affinity="nearest_neighbors", n_neighbors=10)
    model.fit(data.astype("double"))
    pc = PCA(n_components=2).fit_transform(data)
    label = model.fit_predict(data.astype("double"))

    df_pc = pd.DataFrame(zip(pc.T[0].tolist(), pc.T[1].tolist(), label))
    fig = px.scatter(df_pc, x=0, y=1, color=2)
    fig.show()



def plot_clusters_on_pc_kmeans(number_of_clusters, data):
    nbr_clusters = number_of_clusters
    kmeans = KMeans(init="k-means++", n_clusters=nbr_clusters, n_init=4)
    kmeans.fit(data.astype("double"))
    pc = PCA(n_components=2).fit_transform(data)
    # pc = data
    # label = kmeans.fit_predict(pc)
    label = kmeans.fit_predict(data.astype("double"))

    df_pc = pd.DataFrame(zip(pc.T[0].tolist(), pc.T[1].tolist(), label))

    fig = px.scatter(df_pc, x=0, y=1, color=2)
    fig.show()


def plot_clusters_on_pc_spectral_3d(number_of_clusters, data, marker_size=0.5):
    """
    Perform Spectral Clustering and visualize the clusters in 3D using PCA.

    Parameters:
    - number_of_clusters (int): The number of clusters to generate.
    - data (array-like): The dataset to be clustered and visualized.
    - marker_size (float): Size of the markers in the scatter plot.
    """
    nbr_clusters = number_of_clusters
    model = SpectralClustering(
        n_clusters=nbr_clusters,
        assign_labels="discretize",
        random_state=0,
        affinity="nearest_neighbors",
        n_neighbors=10,
    )
    model.fit(data.astype("double"))
    # Increase the number of components for PCA to 3 for 3D visualization
    pc = PCA(n_components=3).fit_transform(data)
    label = model.fit_predict(data.astype("double"))

    # Create a DataFrame with the 3D coordinates and the labels
    df_pc = pd.DataFrame(
        zip(pc[:, 0], pc[:, 1], pc[:, 2], label), columns=["x", "y", "z", "label"]
    )

    # Use Plotly Express to create a 3D scatter plot
    fig = px.scatter_3d(df_pc, x="x", y="y", z="z", color="label")
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(width=1000, height=800)
    fig.show()


def plot_clusters_on_pc_kmeans_3d(number_of_clusters, data, marker_size=0.5):
    nbr_clusters = number_of_clusters
    kmeans = KMeans(init="k-means++", n_clusters=nbr_clusters, n_init=4)
    kmeans.fit(data.astype("double"))
    # Adjust the number of PCA components to 3 for 3D visualization
    pc = PCA(n_components=3).fit_transform(data)
    label = kmeans.fit_predict(data.astype("double"))

    # Create a DataFrame with the 3D coordinates and the labels
    df_pc = pd.DataFrame(
        zip(pc[:, 0], pc[:, 1], pc[:, 2], label), columns=["x", "y", "z", "label"]
    )

    # Use Plotly Express to create a 3D scatter plot
    fig = px.scatter_3d(df_pc, x="x", y="y", z="z", color="label")
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(width=1000, height=800)
    fig.show()


def visualize_main_words_in_clusters_TFIDF(number_of_clusters, data, df_t):
    """
    Performs spectral clustering on the dataset and visualizes the main words in each cluster
    using TF-IDF scores. This visualization is accomplished via a treemap and is made interactive
    through a Dash application. The main words are determined by their TF-IDF scores within
    the text data associated with each cluster.

    Parameters:
    - number_of_clusters (int): Number of clusters for the spectral clustering.
    - data (DataFrame or ndarray): Numerical data for clustering.
    - df_t (DataFrame): DataFrame containing text data corresponding to `data` for TF-IDF analysis.

    The function clusters the `data`, applies PCA for dimensionality reduction, and then uses TF-IDF
    to analyze the text data `df_t`. It visualizes significant words for each cluster in a treemap
    and deploys a Dash app for interactive exploration.
    """
    
    nbr_clusters = number_of_clusters
    model = SpectralClustering(
        n_clusters=nbr_clusters,
        assign_labels="discretize",
        random_state=0,
        affinity="nearest_neighbors",
        n_neighbors=10,
    )
    model.fit(data.astype("double"))
    pc = PCA(n_components=2).fit_transform(data)
    # pc = data#
    # label = model.fit_predict(pc)
    label = model.fit_predict(data.astype("double"))

    df_pc = pd.DataFrame(
        zip(
            pc.T[0].tolist(), pc.T[1].tolist(), label, df_t["text"].apply(read).tolist()
        )
    )

    # df_pc[3] = df_pc[3].apply(word_tokenize)
    df_pc = df_pc.rename(columns={2: "cluster", 3: "text", 0: "pc1", 1: "pc2"})

    df_group = df_pc.groupby(["cluster"]).sum()[["text"]]
    df_group.reset_index(inplace=True)

    conditions = []

    for i in range(nbr_clusters):
        conditions.append((df_group["cluster"] == i))

    values = ["Cluster " + str(i) for i in range(nbr_clusters)]

    df_group["description_cluster"] = np.select(conditions, values)

    colors = [
        "aggrnyl",
        "agsunset",
        "algae",
        "amp",
        "armyrose",
        "balance",
        "blackbody",
        "bluered",
        "blues",
        "blugrn",
        "bluyl",
        "brbg",
        "brwnyl",
        "bugn",
        "bupu",
        "burg",
        "burgyl",
        "cividis",
        "curl",
        "darkmint",
        "deep",
        "delta",
        "dense",
        "earth",
        "edge",
        "electric",
        "emrld",
        "fall",
        "geyser",
        "gnbu",
        "gray",
        "greens",
        "greys",
        "haline",
        "hot",
        "hsv",
        "ice",
        "icefire",
        "inferno",
        "jet",
        "magenta",
        "magma",
        "matter",
        "mint",
        "mrybm",
        "mygbm",
        "oranges",
        "orrd",
        "oryel",
        "oxy",
        "peach",
        "phase",
        "picnic",
        "pinkyl",
        "piyg",
        "plasma",
        "plotly3",
        "portland",
        "prgn",
        "pubu",
        "pubugn",
        "puor",
        "purd",
        "purp",
        "purples",
        "purpor",
        "rainbow",
        "rdbu",
        "rdgy",
        "rdpu",
        "rdylbu",
        "rdylgn",
        "redor",
        "reds",
        "solar",
        "spectral",
        "speed",
        "sunset",
        "sunsetdark",
        "teal",
        "tealgrn",
        "tealrose",
        "tempo",
        "temps",
        "thermal",
        "tropic",
        "turbid",
        "turbo",
        "twilight",
        "viridis",
        "ylgn",
        "ylgnbu",
        "ylorbr",
        "ylorrd",
    ]

    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform(df_group["text"])

    def display_topic_cluster(n):
        df_text_bow = tfidf_matrix.toarray()
        bow_df = pd.DataFrame(df_text_bow)
        bow_df.columns = vectorizer.get_feature_names_out()
        word_freq = pd.DataFrame(
            bow_df[bow_df.index == n].sum().sort_values(ascending=False)
        )
        word_freq.reset_index(level=0, inplace=True)
        word_freq.columns = ["word", "frequency"]

        if n > 0:
            word_freq.drop(index=[0], inplace=True)

        fig = px.treemap(
            word_freq[0:20],
            path=[px.Constant(values[n]), "word"],
            values="frequency",
            color="frequency",
            hover_data=["frequency"],
            color_continuous_scale=colors[n],
        )

        fig.update_layout(autosize=False, width=1000, height=500)

        return fig

    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    # the style arguments for the sidebar.
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "10%",
        "padding": "20px 10px",
        "background-color": "#f8f9fa",
    }

    # the style arguments for the main content page.
    CONTENT_STYLE = {"margin-left": "5%", "margin-right": "5%", "padding": "20px 10p"}

    TEXT_STYLE = {"textAlign": "center", "color": "#191970"}

    CARD_TEXT_STYLE = {"textAlign": "center", "color": "#0074D9"}

    content_first_row = dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id="dropdown",
                    options=[
                        {"label": str(i), "value": i} for i in range(nbr_clusters)
                    ],
                    value=0,
                    clearable=False,
                ),
                md=12,
            )
        ]
    )

    # Adjusted callback structure
    @app.callback(Output("graph", "figure"), [Input("dropdown", "value")])
    def update_graph(selected_cluster):
        # Add callback logic here
        return display_topic_cluster(selected_cluster)

    content_second_row = dbc.Row([dbc.Col(dcc.Graph(id="graph"), md=12)])

    content_zero_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    id="dataset_title_1",
                                    children=["Text Clustering Analysis"],
                                    className="card-title",
                                    style=CARD_TEXT_STYLE,
                                ),
                                html.P(
                                    id="card_text_1",
                                    children=["TITLE"],
                                    style=CARD_TEXT_STYLE,
                                ),
                            ]
                        )
                    ]
                ),
                md=12,
            )
        ]
    )

    """content_third_row = dbc.Row(
    [
        dbc.Col(  
            dcc.Graph(id="graph2", 
                    figure = px.sunburst(df_group, path=['description_cluster'], values='tot',color='cluster')),md=6 ),
        dbc.Col(  
            dcc.Graph(id="graph3", figure =px.scatter(df_group, x="project", y="cluster", size="tot", color="description_cluster").update_layout(legend=dict(
        orientation="h",
        x=0,
        y=1.2,
        yanchor="bottom",
        xanchor="right"
    )) ), md=6)
        ]
    )"""

    content = html.Div(
        [
            # html.H2('Text Analytics Dashboard', style=TEXT_STYLE),
            # html.Hr(),
            content_zero_row,
            html.Br(),
            content_first_row,
            content_second_row,
            # content_third_row,
            html.Hr(),
        ],
        style=CONTENT_STYLE,
    )

    app.layout = html.Div([content])

    app.run_server(debug=False)


def plot_silhouette_and_sse(rank, data):
    print(silhouette_score_([i for i in range(2, rank)], data))
    print(sse_scaler_([i for i in range(2, rank)], data))


def read(text):
    return text.replace("_", " ")
