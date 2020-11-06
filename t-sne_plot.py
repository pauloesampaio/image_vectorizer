import numpy as np
from image_vectorizer.utils import load_config
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models import HoverTool


config = load_config()
xy = np.load(config["tsne_path"])
paths_dataframe = pd.read_csv(config["file_list_path"])


# st.write(
#     """
# # Vanilla image matcher
# ## Enter the image url
# """
# )


def build_plot(xy):
    source = ColumnDataSource(
        data=dict(x=xy[:, 0], y=xy[:, 1], imgs=paths_dataframe["filename"].to_list())
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
    graph.scatter("x", "y", source=source, alpha=0.5, size=5)
    return graph


fig = build_plot(xy)
show(fig)
