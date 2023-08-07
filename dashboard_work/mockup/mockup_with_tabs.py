from pathlib import Path

import pandas as pd
import panel as pn
from panel.pane import VTK

pn.extension()

pn.extension('vtk')
pn.extension("mathjax", sizing_mode="stretch_width", template="bootstrap")


import panel as pn
import hvplot.pandas

# Load Data
from bokeh.sampledata.autompg import autompg_clean as df


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

markdown_pane = pn.pane.Markdown(r"""
# Title

## Header

### Sub Header

```python
import panel as pn

pn.extension()
```

---

$$ a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} = \sum_{k=0}^{n-1} ar^k = a \left(\frac{1-r^{n}}{1-r}\right)$$
""")

import random

rcolor = lambda: "#%06x" % random.randint(0, 0xFFFFFF)

box = pn.FlexBox(*[pn.pane.HTML(str(i), styles=dict(background=rcolor()), width=100, height=100) for i in range(24)])


# vtk = VTK(
#    'https://raw.githubusercontent.com/Kitware/vtk-js/master/Data/StanfordDragon.vtkjs',
#     sizing_mode='stretch_width', height=400, enable_keybindings=True,
#     orientation_widget=True
# )


text = (
    "This is a **{alert_type}** alert with [an example link]"
    "(https://panel.holoviz.org/). Give it a click if you like."
)

pn.Column(*[
    pn.pane.Alert(text.format(alert_type=at), alert_type=at)
    for at in pn.pane.Alert.param.alert_type.objects],
    sizing_mode="stretch_width"
).servable()

alerts = pn.Column(
    pn.pane.Alert("DeprecationWarning: The truth value of an empty array is ambiguous. Returning"
               "False, but in future this will result in an error. Use `array.size > 0` to check "
               "that an array is not empty.", alert_type="warning"),
    pn.pane.Alert("Found 10 unexpected evaluation errors in IPOPT.out", alert_type="danger"),
    pn.pane.Alert("'dupcomp' <class DupPartialsComp>: d(x)/d(c): declare_partials has been called with rows and cols that specify the following duplicate subjacobian entries: [(4, 11), (10, 2)].",
                  alert_type="primary"),
)

df = pd.DataFrame({ 'varname': ['x', 'y', 'z'], 'value': [3.14, 6.28, 9.42],'units': ['m', 'kg', 'sec']})
results_pane = pn.widgets.DataFrame(df, name='Results')

with open("assets/opt_report.html","r") as f:
    opt_report_html = f.read()
opt_report = pn.pane.HTML(opt_report_html)

driver_scaling_report = pn.pane.HTML('<iframe width=1000 height=1000 src=assets/driver_scaling_report.html></iframe>', sizing_mode='stretch_width')


tabs = pn.Tabs(
    ('Model Basics', markdown_pane),
    ('N2', pn.FlexBox(html_pane)),
    ('Driver Scaling Report', driver_scaling_report),
    ('3D model', plane_pane),
    ('Opt Report', opt_report),
    ('Optimization Plot', p2),
    ('Alerts', alerts),
    ('Results Table', results_pane),
    )
# tabs = pn.Tabs(('Tab 1', p1), p2, box)


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
