import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def histogram(df: pd.DataFrame, column: str, nbins: int = 30):
    return px.histogram(df, x=column, nbins=nbins, title=f'Histogram of {column}')


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]):
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title='Correlation matrix')
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    return px.scatter(df, x=x, y=y, color=color, title=f'{y} vs {x}')
