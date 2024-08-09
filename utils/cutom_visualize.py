import itertools
import numpy as np
import pandas as pd
from umap import UMAP
from typing import List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_barchart(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = 8,
    n_words: int = 5,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topic Word Scores</b>",
    width: int = 250,
    height: int = 250,
    autoscale: bool = False,
    repr_name: str = "default_representation"
) -> go.Figure:
    """Visualize a barchart of selected topics.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.
        autoscale: Whether to automatically calculate the height of the figures to fit the whole bar text
        repr_name: The name of the representation to use for grabbing keywords.

    Returns:
        fig: A plotly figure

    Examples:
    To visualize the barchart of selected topics
    simply run:

    ```python
    topic_model.visualize_barchart()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=False,
        horizontal_spacing=0.1,
        vertical_spacing=0.4 / rows if rows > 1 else 0,
        subplot_titles=subplot_titles,
    )

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        try:
            topic_info = topic_model.get_topic(topic, full=True)[repr_name]
        except KeyError:
            raise KeyError(f"Representation '{repr_name}' does not exist. "
                           f"Choose one of the following: {list(topic_model.get_topics().keys())}")
        words = [word + "  " for word, _ in topic_info][:n_words][::-1]
        scores = [score for _, score in topic_info][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores, y=words, orientation="h", marker_color=next(colors)),
            row=row,
            col=column,
        )

        if autoscale:
            if len(words) > 12:
                height = 250 + (len(words) - 12) * 11

            if len(words) > 9:
                fig.update_yaxes(tickfont=dict(size=(height - 140) // len(words)))

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def visualize_documents_per_class(
    topic_model,
    docs: List[str],
    classes: List[str],
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    sample: float = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Documents and Classes</b>",
    width: int = 1200,
    height: int = 750,
    selected_classes: List[str] = None,
):
    """Visualize documents and their classes in 2D.

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        classes: The class labels for each document.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each class that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each class) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualization.
        custom_labels: If bool, whether to use custom class labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.
        selected_classes: List of classes to be highlighted. If None, all classes are highlighted.
    """
    class_per_doc = classes

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for class_ in set(class_per_doc):
        s = np.where(np.array(class_per_doc) == class_)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"class": np.array(class_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["class"] = [class_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine").fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_classes = set(class_per_doc)

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(cls), None]] + topic_model.topic_aspects_[custom_labels][cls] for cls in unique_classes]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[cls + topic_model._outliers] for cls in unique_classes]
    else:
        names = list(unique_classes)

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected classes
    if selected_classes is None:
        selected_classes = unique_classes

    non_selected_classes = set(unique_classes).difference(selected_classes)
    if len(non_selected_classes) == 0:
        non_selected_classes = [-1]

    selection = df.loc[df["class"].isin(non_selected_classes), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [
        None,
        None,
        selection.x.mean(),
        selection.y.mean(),
        "Other documents",
    ]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode="markers+text",
            name="other",
            showlegend=False,
            marker=dict(color="#CFD8DC", size=5, opacity=0.5),
        )
    )

    # Selected classes
    for name, cls in zip(names, unique_classes):
        if cls in selected_classes:
            selection = df.loc[df["class"] == cls, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [
                    None,
                    None,
                    selection.x.mean(),
                    selection.y.mean(),
                    name,
                ]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode="markers+text",
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5),
                )
            )

    # Add grid in a 'plus' shape
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig