"""Streaming is something that Panel supports really well

This application demonstrates how to use
[Streamz](https://streamz.readthedocs.io/en/latest/),
[Pandas](https://pandas.pydata.org/),
[Bokeh](https://docs.bokeh.org/en/latest/index.html),
[Holoviews](https://holoviews.org/),
[Altair](https://altair-viz.github.io/),
[Echart](https://echarts.apache.org/en/index.html) and
[Plotly](https://plotly.com/) for streaming.

The stream will STOP AFTER 1 MINUTE.
"""
from datetime import datetime

import altair as alt
import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
from streamz.dataframe import DataFrame as sDataFrame

from awesome_panel import config

alt.themes.enable("dark")

ACCENT = config.ACCENT
THEME = {
    pn.template.theme.DarkTheme: {pn.pane.ECharts: "dark", pn.pane.Plotly: "plotly_dark"},
    pn.template.theme.DefaultTheme: {pn.pane.ECharts: "default", pn.pane.Plotly: "plotly_white"},
}

HEADER = [config.get_header()]
SIDEBAR_FOOTER = config.menu_fast_html(accent=ACCENT)


def _create_echarts(data):
    data = data.sort_index()
    return {
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": list(zip(list(data.index.values), list(data["y"]))),
                "type": "line",
                "showSymbol": False,
                "hoverAnimation": False,
                "itemStyle": {"color": ACCENT},
            },
        ],
        "responsive": True,
    }


def _create_altair(data):
    data = data.reset_index()
    return (
        alt.Chart(data, width="container")
        .mark_line()
        .encode(
            x="index",
            y="y",
        )
    )


def _create_hvplot(data):
    # Hack:
    # hv.renderer("bokeh").theme = pn.template.react.DarkTheme.param.bokeh_theme.default
    return data.hvplot(y="y", color=ACCENT).opts(default_tools=[])


def _create_plotly(data, template="plotly_white"):
    plot = px.line(data, y="y", template=template)
    plot.layout.autosize = True
    # plot.layout.plot_bgcolor = "#2a2a2a"
    # plot.layout.paper_bgcolor = "#2a2a2a"
    plot.layout.xaxis.tickformat = "%H:%M:%S"
    return plot


def view(intro_section):
    """Returns an instance of the Streaming Plots app"""
    pn.config.sizing_mode = "stretch_width"
    template = pn.template.FastGridTemplate(
        theme="dark",
        main_max_width="",
        row_height=110,
        accent_base_color=ACCENT,
        header_background=ACCENT,
        prevent_collision=True,
        save_layout=True,
        sidebar_footer=SIDEBAR_FOOTER,
        header=HEADER,
    )
    echart_pane = pn.pane.ECharts(
        theme=THEME[template.theme][pn.pane.ECharts], sizing_mode="stretch_both"
    )
    plotly_pane = pn.pane.Plotly(
        sizing_mode="stretch_width", height=371, config={"responsive": True}
    )
    hvplot_pane = pn.pane.HoloViews(sizing_mode="stretch_both")
    # Cannot be stretch_both due to
    # https://github.com/holoviz/panel/issues/4034
    altair_pane = pn.pane.Vega(sizing_mode="stretch_both", height=400)

    df_stream = sDataFrame(example=pd.DataFrame({"y": []}, index=pd.DatetimeIndex([])))
    df_window_stream = df_stream.cumsum().stream.sliding_window(50).map(pd.concat)

    def update_echarts(data):
        echart_pane.object = _create_echarts(data)

    def update_plotly(data):
        plotly_pane.object = _create_plotly(data, template=THEME[template.theme][pn.pane.Plotly])

    def update_altair(data):
        altair_pane.object = _create_altair(data)

    def update_hvplot(data):
        hvplot_pane.object = _create_hvplot(data)

    df_window_stream.sink(update_echarts)
    df_window_stream.sink(update_plotly)
    df_window_stream.sink(update_altair)
    df_window_stream.sink(update_hvplot)

    def emit():
        df_stream.emit(
            pd.DataFrame({"y": [np.random.randn()]}, index=pd.DatetimeIndex([datetime.utcnow()]))
        )

    emit()
    pn.state.add_periodic_callback(emit, period=250, count=240)  # Will stream for 1 mins

    template.main[0:3, 0:12] = intro_section
    template.main[3:7, 0:6] = pn.Column(
        pn.pane.Markdown("## BOKEH via HOLOVIEWS"), hvplot_pane, sizing_mode="stretch_both"
    )
    template.main[3:7, 6:12] = pn.Column(
        pn.pane.Markdown("## VEGA via ALTAIR"), altair_pane, sizing_mode="stretch_both"
    )
    template.main[7:11, 0:6] = pn.Column(
        pn.pane.Markdown("## PLOTLY"), plotly_pane, sizing_mode="stretch_both"
    )
    template.main[7:11, 6:12] = pn.Column(
        pn.pane.Markdown("## ECHART"), echart_pane, sizing_mode="stretch_both"
    )
    return template


def serve():
    """Serves the app"""
    app = config.extension(
        "plotly", "vega", "echarts", url="streaming_plots", template=None, intro_section=None
    )
    intro_section = app.intro_section()

    view(intro_section=intro_section).servable()


if __name__.startswith("bokeh"):
    serve()
