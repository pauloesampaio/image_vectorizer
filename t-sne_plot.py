import numpy as np
from image_vectorizer.utils import load_config
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models import HoverTool
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from sklearn.manifold import TSNE


def calculate_tsne(vectors):
    """Calculates 2-dimensional t-SNE representation of given set of vectors

    Args:
        vectors (np.array): Set of vectors

    Returns:
        np.array: 2-dim t-SNE representation
    """
    tsne = TSNE(random_state=12345, verbose=2, n_jobs=-1)
    xy = tsne.fit_transform(vectors)
    return xy


def build_plot(xy, paths, labels=None):
    """Builds t-SNE scatterplot

    Args:
        xy (np.array): Set of vectors
        paths (list): list of paths
        labels (bool): Bool, if labels should be used or not

    Returns:
        bokeh.graph: bokeh plot

    """
    if labels:
        colors = factor_cmap("labels", palette=Spectral6, factors=list(set(labels)))
        plot_labels = labels
    else:
        colors = factor_cmap("labels", palette=Spectral6, factors=["None"])
        plot_labels = ["None"] * len(xy)

    source = ColumnDataSource(
        data=dict(
            x=xy[:, 0],
            y=xy[:, 1],
            imgs=paths,
            labels=plot_labels,
        ),
    )
    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="96"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
                <span style="font-size: 10px;">Label: "@labels"</span>
            </div>
        </div>
        """
    )
    graph = figure(
        tools=[hover, "box_select"],
        plot_width=640,
        plot_height=640,
        title="t-SNE plot (hover for images)",
    )
    if labels:
        graph.scatter(
            "x",
            "y",
            source=source,
            alpha=0.5,
            size=5,
            color=colors,
            legend_group="labels",
        )
    else:
        graph.scatter("x", "y", source=source, alpha=0.5, size=5)

    return graph


if __name__ == "__main__":
    config = load_config()
    vectors = np.load(config["vectors_path"])
    xy = calculate_tsne(vectors)
    paths_dataframe = pd.read_csv(config["file_list_path"])
    paths = paths_dataframe["filename"].tolist()
    if config["infer_classes"]:
        labels = paths_dataframe["class"].tolist()
    else:
        labels = None

    fig = build_plot(xy, paths, labels)
    show(fig)
