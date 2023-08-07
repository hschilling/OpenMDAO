from pathlib import Path

import pandas as pd
import panel as pn
import openmdao.api as om
import numpy as np

pn.extension()
pn.extension('vtk')
pn.extension("mathjax", sizing_mode="stretch_width", template="bootstrap")
pn.extension('tabulator')


import panel as pn
import hvplot.pandas

# Load Data
from bokeh.sampledata.autompg import autompg_clean as df


filename = "sellar_recording.sql"
cr = om.CaseReader(filename)
driver_cases = cr.list_cases('driver', out_stream=None)
df = pd.DataFrame()
for i, iter_coord in enumerate(driver_cases):
    case = cr.get_case(iter_coord)
    for key, value in case.outputs.items():
        if key not in df.columns:
            df[key] = None  # None represents the initial values in the column
        df.loc[i, key] = np.linalg.norm(value)

df.to_csv("sellar_recording_data.csv")
def pipeline(df=df):
    return (
        df
    )
line_width = pn.widgets.IntSlider(value=6, start=1, end=10, name="Line Width")
variables = pn.widgets.CheckBoxGroup(
    name="Variables", options=list(df.columns), value=list(df.columns)
)
ipipeline = pipeline(df.interactive())
PALETTE = ["#ff6f69", "#ffcc5c", "#88d8b0", ]
ACCENT_BASE_COLOR = PALETTE[0]

from bokeh.palettes import Category10
ihvplot = ipipeline.hvplot(y=variables,responsive=True,min_height=400, color=list(Category10[10]),

# ihvplot = ipipeline.hvplot(y=variables,responsive=True, min_height=400, color=PALETTE,
                           line_width=line_width, yformatter="%.0f",
                           title="Model Optimization using OpenMDAO")
ihead = ipipeline.head()



plot_size = 1000

from bokeh.plotting import figure

p1 = figure(width=plot_size, height=plot_size, name='Plot1')
p1.scatter([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 2, 1, 0])

p2 = figure(width=plot_size, height=plot_size, name='Line')
p2.line([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 2, 1, 0])

with open("n2.html","r") as f:
    n2_file_html = f.read()

# html_pane = pn.pane.HTML(n2_file_html)

# html_pane = pn.pane.HTML('<iframe width=100% height=400 src=https://openmdao.org/newdocs/versions/latest/features/model_visualization/n2_basics/n2_basics.html?></iframe>', sizing_mode='stretch_width')
# html_pane = pn.pane.HTML('<iframe width=100% height=400 src=/Users/hschilli/Documents/OpenMDAO/dev/panel_with_clean_env/dashboard_work/mockup/n2.html></iframe>', sizing_mode='stretch_width')
# html_pane = pn.pane.HTML(f'<iframe width=100% height=400 srcdoc={n2_file_html}/></iframe>', sizing_mode='stretch_width')
html_pane = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/n2.html></iframe>', sizing_mode='stretch_width')


# https://replit.com/@SixBeeps/Beach#index.html
plane_pane = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/plane_model.html></iframe>', sizing_mode='stretch_width')

model_basics_tab = pn.pane.Markdown(r"""
# Model Basics

## Design Variables

| Name    | Initial Value | Shape | Units |
| ------- | ------------- | ----- | ----- |
| x       | 3.0           | (1,)  | None  |
| z       | [1., 2.]      | (2,)  | None  |

## Constraints

| Name    | Initial Value | Shape | Units |
| ------- | ------------- | ----- | ----- |
| con1    | 0.0           | (1,)  | None  |
| con2    | 0.0           | (1,)  | None  |

## Objective

| Name    | Initial Value | Shape | Units |
| ------- | ------------- | ----- | ----- |
| obj     | 0.0           | (1,)  | None  |
""")

import random

rcolor = lambda: "#%06x" % random.randint(0, 0xFFFFFF)

box = pn.FlexBox(*[pn.pane.HTML(str(i), styles=dict(background=rcolor()), width=100, height=100) for i in range(24)])


# vtk = VTK(
#    'https://raw.githubusercontent.com/Kitware/vtk-js/master/Data/StanfordDragon.vtkjs',
#     sizing_mode='stretch_width', height=400, enable_keybindings=True,
#     orientation_widget=True
# )


# text = (
#     "This is a **{alert_type}** alert with [an example link]"
#     "(https://panel.holoviz.org/). Give it a click if you like."
# )
#
# pn.Column(*[
#     pn.pane.Alert(text.format(alert_type=at), alert_type=at)
#     for at in pn.pane.Alert.param.alert_type.objects],
#     sizing_mode="stretch_width"
# ).servable()

alerts = pn.Column(
    pn.pane.Markdown('''
# Alerts, Warnings, and Errors    
    '''),
    pn.pane.Alert("DeprecationWarning: The truth value of an empty array is ambiguous. Returning"
               "False, but in future this will result in an error. Use `array.size > 0` to check "
               "that an array is not empty.", alert_type="warning"),
    pn.pane.Alert("Found 10 unexpected evaluation errors in IPOPT.out", alert_type="danger"),
    pn.pane.Alert("'dupcomp' <class DupPartialsComp>: d(x)/d(c): declare_partials has been called with rows and cols that specify the following duplicate subjacobian entries: [(4, 11), (10, 2)].",
                  alert_type="primary"),
)

df = pd.DataFrame({ 'varname': ['x', 'z', 'con1', 'con2', 'obj'],
                    'value': [2.0, [1.97763888e+00, 1.57073826e-15], -8.58912941e-11, -20.24472223, 3.18339395],
                    'units': ['m', 'm', 'None', 'None', 'kg']})
# results_pane = pn.widgets.DataFrame(df, name='Results',autosize_mode='fit_columns', width=500)
variables_filters = {
    'varname': {'type': 'input', 'func': 'like', 'placeholder': 'Enter varname'},
    # 'Year': {'placeholder': 'Enter year'},
}

results_pane = pn.widgets.Tabulator(df, layout='fit_data', show_index=False, header_align='left', text_align='left',
                                    header_filters=variables_filters,
                                    width=1000, height=1000
                                    )


with open("assets/opt_report.html","r") as f:
    opt_report_html = f.read()
opt_report = pn.pane.HTML(opt_report_html)

driver_scaling_report = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/driver_scaling_report.html></iframe>', sizing_mode='stretch_width')

# plot_column = pn.Column(
#     ihead.panel(),
#     pn.Row(
#     line_width,
#     variables),
#     ihvplot.panel(),
# )


plot_column = pn.Column(
                pn.Row(
                    pn.Column(
                        variables,
                        pn.VSpacer(height=30),
                        line_width,
                        pn.VSpacer(height=30),
                        width = 300
                    ),
                    ihvplot.panel(),
                )
               )

tabs = pn.Tabs(
    ('Model Basics', model_basics_tab),
    ('N2', pn.Column(pn.pane.Markdown('# N2 Diagram'), html_pane)),
    ('Driver Scaling Report', pn.Column( pn.pane.Markdown('# Driver Scaling Report'),driver_scaling_report)),
    ('3D model', pn.Column( pn.pane.Markdown('# Driver Scaling Report'),plane_pane)),
    ('Opt Report', opt_report),
    # ('Optimization Plot', p2),
    ('Optimization Plot', pn.Column( pn.pane.Markdown('# Optimization Plot'),plot_column)),
    ('Alerts', alerts),
    ('Results Table', pn.Column( pn.pane.Markdown('# Results Table'),results_pane)),
    )


# Layout using Template
# template = pn.template.FastListTemplate(
#     title='Interactive DataFrame Dashboards with hvplot .interactive',
#     sidebar=[cylinders, 'Manufacturers', mfr, 'Y axis' , yaxis],
#     main=[ihvplot.panel()],
#     accent_base_color="#88d8b0",
#     header_background="#88d8b0",
# )

template = pn.template.FastListTemplate(
# template = pn.template.VanillaTemplate(
# template = pn.template.ReactTemplate(
# template = pn.template.MaterialTemplate(
# template = pn.template.GoldenTemplate(
# template = pn.template.BootstrapTemplate(
    title='Aviary Dashboard',
    main=[tabs],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)
from bokeh.resources import INLINE
# template.save('export.html')
# tabs.save('export.html', resources=INLINE)

#
# tabs.servable()

import bokeh
print(bokeh.util.paths.bokehjsdir())


# template.servable()

# pn.serve({'app11':get_page1},show=True,autoreload=True,session_history=-1,
# port=5007,address='147.47.240.86',host="0.0.0.0",allow_websocket_origin=['*'],
# rest_session_info=True,auth_module='auth.py', log_file='log_panel')


if __name__ == "__main__":
    # pn.serve({Path(__file__).name: tabs})
    pn.serve(template, static_dirs={'assets': './assets'})
# if __name__ == '__main__':
#     from bokeh.server.server import Server
#
#     server = pn.serve(template, port=5006, allow_websocket_origin=["localhost:5006"], show=False, start=False)
#     print(server)
#     server.start()
#     from bokeh.io import curdoc, show
#     show(server)
#     # server.show("/")
