import panel as pn
import hvplot.pandas

# Load Data
from bokeh.sampledata.autompg import autompg_clean as df

# Make DataFrame Pipeline Interactive
idf = df.interactive()

# Define Panel widgets
cylinders = pn.widgets.IntSlider(name='Cylinders', start=4, end=8, step=2)
mfr = pn.widgets.ToggleGroup(
    name='MFR',
    options=['ford', 'chevrolet', 'honda', 'toyota', 'audi'],
    value=['ford', 'chevrolet', 'honda', 'toyota', 'audi'],
    button_type='success')
yaxis = pn.widgets.RadioButtonGroup(
    name='Y axis',
    options=['hp', 'weight'],
    button_type='success'
)

# Combine pipeline and widgets
ipipeline = (
    idf[
        (idf.cyl == cylinders) &
        (idf.mfr.isin(mfr))
    ]
    .groupby(['origin', 'mpg'])[yaxis].mean()
    .to_frame()
    .reset_index()
    .sort_values(by='mpg')
    .reset_index(drop=True)
)

# Pipe to hvplot
ihvplot = ipipeline.hvplot(x='mpg', y=yaxis, by='origin', color=["#ff6f69", "#ffcc5c", "#88d8b0"], line_width=6, height=400)

# Layout using Template
template = pn.template.FastListTemplate(
    title='Interactive DataFrame Dashboards with hvplot .interactive',
    sidebar=[cylinders, 'Manufacturers', mfr, 'Y axis' , yaxis],
    main=[ihvplot.panel()],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)
template.servable()