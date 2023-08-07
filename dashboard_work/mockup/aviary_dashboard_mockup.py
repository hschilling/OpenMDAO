import pandas as pd
import panel as pn
from bokeh.palettes import Category10
import hvplot.pandas  # noqa # need this ! Otherwise hvplot using DataFrames does not work

pn.extension()
pn.extension("mathjax", sizing_mode="stretch_width", template="bootstrap") # if using an math eqs
pn.extension('tabulator')

#### Model Basics ####
model_basics_pane = pn.pane.Markdown(r"""
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

####  N2  ####
n2_pane = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/n2.html></iframe>', sizing_mode='stretch_width')

#### Driver Scaling Report ####
driver_scaling_report_pane = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/driver_scaling_report.html></iframe>',
                                          sizing_mode='stretch_width')

#### aircraft Model ####
aircraft_pane = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/plane_model.html></iframe>',
                             sizing_mode='stretch_width')

#### Optimizer report ####
with open("assets/opt_report.html","r") as f:
    opt_report_html = f.read()
opt_report_pane = pn.pane.HTML(opt_report_html)

#### Optimization plot ####
df = pd.read_csv('sellar_recording_data.csv', index_col=0)
line_width = pn.widgets.IntSlider(value=6, start=1, end=10, name="Line Width")
variables = pn.widgets.CheckBoxGroup(
    name="Variables", options=list(df.columns), value=list(df.columns)
)
# ipipeline = pipeline(df.interactive())
ipipeline = df.interactive()
ihvplot = ipipeline.hvplot(y=variables,responsive=True,min_height=400, color=list(Category10[10]),
                           line_width=line_width, yformatter="%.0f",
                           title="Model Optimization using OpenMDAO")
optimization_plot_pane = pn.Column(
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

#### Alerts ####
alerts_pane = pn.Column(
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

#### Results pane ####
df = pd.DataFrame({ 'varname': ['x', 'z', 'con1', 'con2', 'obj'],
                    'value': [2.0, [1.97763888e+00, 1.57073826e-15], -8.58912941e-11, -20.24472223, 3.18339395],
                    'units': ['m', 'm', 'None', 'None', 'kg']})
variables_filters = {
    'varname': {'type': 'input', 'func': 'like', 'placeholder': 'Enter varname'},
}
results_pane = pn.widgets.Tabulator(df, layout='fit_data', show_index=False, header_align='left', text_align='left',
                                    header_filters=variables_filters,
                                    width=1000, height=1000
                                    )

#### Collect all the Tabs ####
tabs = pn.Tabs(
        ('Model Basics', model_basics_pane),
        ('N2', pn.Column(pn.pane.Markdown('# N2 Diagram'), n2_pane)),
        ('Driver Scaling Report', pn.Column( pn.pane.Markdown('# Driver Scaling Report'),driver_scaling_report_pane)),
        ('3D model', pn.Column( pn.pane.Markdown('# Driver Scaling Report'),aircraft_pane)),
        ('Opt Report', opt_report_pane),
        ('Optimization Plot', pn.Column( pn.pane.Markdown('# Optimization Plot'),optimization_plot_pane)),
        ('Alerts', alerts_pane),
        ('Results Table', pn.Column( pn.pane.Markdown('# Results Table'),results_pane)),
    )

template = pn.template.FastListTemplate(
    title='Aviary Dashboard',
    main=[tabs],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)

if __name__ == "__main__":
    pn.serve(template, static_dirs={'assets': './assets'})
