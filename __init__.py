# modules
import numpy as np
import os, sys, subprocess
import sys, subprocess, os
from IPython.display import display, Javascript, HTML, Math, Markdown

# suppress traitlets futurewarning 
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

# widgets
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, Layout, HBox, Box, VBox, Output
from ipywidgets import IntSlider, FloatSlider, Dropdown, ToggleButton, ToggleButtons, Label
from ipywidgets import HTML as iHTML
import bqplot.pyplot as plt
from bqplot import Axis, LinearScale, HeatMap, ColorAxis, Figure, Lines, ColorScale
from bqplot.interacts import BrushIntervalSelector

# pyunicorn
from pyunicorn.timeseries import RecurrencePlot, CrossRecurrencePlot

# fix for importing pandas and seaborn with numpy 1.24.0
def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x
    return npwarn_decorator

# plotting
try:
    import seaborn as sns, pandas as pd
except:
    # use dummy warnings
    np.linalg._umath_linalg._ilp64 = getattr(np, '_ilp64', dummy_npwarn_decorator_factory)
    np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)
    import seaborn as sns, pandas as pd

# doesn't work
# from scipy.stats import norm 

import plotly
import plotly.graph_objects as go, plotly.express as px
import matplotlib as mpl, matplotlib.pyplot as pplt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
plotly.offline.init_notebook_mode()
    
# import classes
from ._widgets import WidgetBase
from ._plotting import WidgetPlots
from ._timeseries import TimeSeries

class Widget(WidgetBase, WidgetPlots, TimeSeries):

    def __init__(self):
       
        """ Initialize widget instance with styles, settings, and sliders."""
        WidgetBase.__init__(self)
        WidgetPlots.__init__(self)
        TimeSeries.__init__(self)

        self.ts = None
        self.line = None
        self.rp_instance = None
        self.rp_matrix = None
        self.rp_matrix2 = None
        self.ts_plot = None
        self.rp_plot = None
        self.window_selector = None
        self.colors = sns.color_palette('husl').as_hex()
 
        # self.example_rps = None
        # self.stats_formatted = np.array(['𝐷𝐸𝑇', '𝐿𝐴𝑀', '𝐿 𝑀𝐴𝑋', '𝐿 𝑀𝐸𝐴𝑁', '𝑉 𝑀𝐴𝑋', '𝑉 𝑀𝐸𝐴𝑁', '𝐿 𝐸𝑁𝑇𝑅', '𝑉 𝐸𝑁𝑇𝑅', '𝐷𝐼𝑉', '𝑇𝑇'])
        self.stats = np.array(['DET', 'LAM', 'L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'L ENTR', 'V ENTR', 'DIV', 'TT'])
      
        self.style = {'description_width': 'initial', "button_width": "auto", 'overflow':'visible',
            'white-space': 'nowrap', 'button_color': 'lemonchiffon', 'handle_color': 'cornflowerblue',
            'font_family': "serif"}

        self.widget_layout = Layout(width='100%', height='25px', margin="5px", font_family="serif")

        self.button_layout = Layout(width='auto', max_width="4350px", height='25px', margin="10px", font_family="serif")

        self.ui_layout = Layout(display="flex", height="350px", flex_flow="column", overflow="visible",
            align_items="center", justify_content="center", width="45%", font_family="CMU Serif")

        self.horizontal_layout = Layout(display="flex", flex_flow="row", height="350px", overflow= "visible",
            align_items="center", justify_content="center", font_family="serif")

        self.vertical_layout = Layout(display="flex", flex_flow="column", height="350px", overflow= "visible",
            align_items="center", justify_content="center", font_family="serif")

        self.label_layout = Layout(display="flex", flex_flow="row", align_items="center", justify_content="center",
            width="100%", text_align="center", height='40px', margin="10px", font_family="serif")

        self.slider_layout = Layout(min_width="350px", font_family="serif")

        self.dropdown_layout = Layout(max_width="200px", overflow="visible")

        # explicitly set width and height
        self.layout = Layout(width='auto', height='25px', display="flex", margin="10px", font_family="serif")

        # widgets to set parameters of example time series
        self.duration = IntSlider(value=500,min=500,max=1000,step=10, description='𝑡',continuous_update=False)

        self.duration_short = IntSlider(value=150,min=100,max=500,step=10,description='𝑡',continuous_update=False)

        self.time_step = FloatSlider(value=0.01,min=-0.01,max=0.01,step=0.001,
            description='𝑑𝑡',continuous_update=False)

        self.amplitude = FloatSlider(value=1,min=0.1,max=5,step=0.1,description='𝐴',
            continuous_update=False,description_allow_html=True)

        self.frequency = IntSlider(value=2,min=1,max=20,step=1,description='𝜔',
            continuous_update=False,description_allow_html=True)
        
        self.phi = FloatSlider(min=-3.2, max=3.2, step=0.2, value=0.0, description="𝜑")

        self.N = IntSlider(value=350,min=100,max=500,step=10,description='# Steps',continuous_update=False)

        self.y0 = IntSlider(value=5,min=1,max=30,step=1,description='𝑦₀',continuous_update=False)

        # 𝑅𝑉𝑆 to use
        # logistic map parameters
        self.r = FloatSlider(value=3, min=2,max=4,step=0.1,description='𝑟',continuous_update=False)

        self.x = FloatSlider(value=0.617,min=0,max=1,step=0.01,description='𝑥',continuous_update=False)

        self.drift = FloatSlider(value=0.0,min=-0.5,max=0.5,step=0.01,description='Drift',
            continuous_update=False,description_allow_html=True)

        # periodic signals
        self.amp1 = FloatSlider(value=1,min=0.1,max=5,step=0.1,description='𝐴₁',
            continuous_update=False,description_allow_html=True)

        self.freq1 = IntSlider(value=3,min=1,max=20,step=1,description='𝜔₁' ,
            continuous_update=False,description_allow_html=True)

        self.amp2 = FloatSlider(value=1,min=0.1,max=5,step=0.1,description='𝐴₂',
            continuous_update=False,description_allow_html=True)

        self.freq2 = IntSlider(value=3,min=1,max=20,step=1,description='𝜔₂',
            continuous_update=False,description_allow_html=True)

        self.signal1_type = Dropdown(description='Signal 1',values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])

        self.signal2_type = Dropdown(description='Signal 2',values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])

        self.mean = FloatSlider(value=0,min=0,max=5,step=0.1,description='𝜇',
            continuous_update=False,description_allow_html=True)

        self.std = FloatSlider(value=1,min=0.1,max=5,step=0.1,description='𝜎',
            continuous_update=False,description_allow_html=True)

        # self.factor = IntSlider(value=300,min=100,max=1000,step=10,description='Amplification',
        #     continuous_update=False,description_allow_html=True)

        # Rossler bifurcation parameter values
        self.a = FloatSlider(value=0.2,min=-0.5,max=0.55,step=0.01,description='𝑎',
            continuous_update=False,description_allow_html=True)

        self.b = FloatSlider(value=0.2,min=0,max=2,step=0.01,description='𝑏',
            continuous_update=False,description_allow_html=True)

        self.c = FloatSlider(value=5.7,min=0,max=45,step=0.01,description='𝑐',
            continuous_update=False,description_allow_html=True)

        self.signal1 = VBox([self.signal1_type, self.amp1, self.freq1], layout=self.dropdown_layout)
        self.signal2 = VBox([self.signal2_type, self.amp2, self.freq2], layout=self.dropdown_layout)
        self.superimposed = HBox(children=[self.signal1, self.signal2], layout=self.dropdown_layout)

        # set up widgets for RQA
        self.rqa_stat = Dropdown(options=self.stats, layout=self.dropdown_layout)

        self.tau = IntSlider(value=1,min=1,max=10,step=1,continuous_update=False, description="Time Delay (𝜏)",
            style=self.style, layout=self.slider_layout)

        self.dim = IntSlider(value=1,min=1,max=10,step=1,description='Embedding Dimension (𝑚):',
            continuous_update=False, style=self.style, layout=self.slider_layout)

        self.normalize = ToggleButton(description="Normalize time series", value=True,
            continuous_update=False, disabled=False, style=self.style, layout=self.button_layout)

        self.metric = ToggleButtons(options=['Euclidean', 'Manhattan', 'Supremum'],
            continuous_update=False, disabled=False, style=self.style, layout=self.layout)

        self.metric_label = iHTML('Distance Metric:', layout=self.layout)

        self.RR = FloatSlider(value=0.35,min=0.01,max=0.99,step=0.01,description="Recurrence Rate (𝑅𝑅)",
            continuous_update=False, style=self.style, layout=self.slider_layout)

        self.which_RR = ToggleButtons(options=['Global 𝑅𝑅', 'Local 𝑅𝑅'],
            continuous_update=False, style=self.style, layout=self.layout)

        self.which_RR_label = iHTML("For 𝑅𝑄𝐴, threshold by:", layout=self.layout)

        self.threshold = ToggleButton(description='Threshold 𝑅𝑃', value=False,
            continuous_update=False, disabled=False, style=self.style, layout=self.button_layout)

        self.signal = Dropdown(
            description='Time Series:', options=['White Noise', 'Sine', 'Cosine', 'Superimposed', 'Logistic Map', 'Brownian Motion'],
            values=['White Noise', 'Sine', 'Cosine', 'Superimposed', 'Logistic Map', 'Brownian Motion'])

        self.lmin = IntSlider(value=2,min=2,max=10,step=1,description='𝐿 𝑀𝐼𝑁:',
            continuous_update=False, style=self.style, layout=self.slider_layout)

        self.vmin = IntSlider(value=2,min=2,max=10,step=1,description='𝑉 𝑀𝐼𝑁:',
            continuous_update=False, style=self.style, layout=self.slider_layout)

        self.theiler = IntSlider(value=1, min=1, max=20, step=1, description="Theiler Window",
            continuous_update=False, style=self.style, layout=self.slider_layout)

        # initialize user interface with white noise
        # self.ui = Output(layout=Layout(max_height="100px", width="200px", display="flex", overflow="visible"))
        self.ts_ui = VBox([self.duration, self.mean, self.std], layout=Layout(display="flex", overflow="visible"))
        self.threshold_by_rr35 = ToggleButton(description='Threshold by 𝑅𝑅 = 35%', value=False,
            continuous_update=False, disabled=False, style=self.style, layout=self.button_layout)
        # Add the 'set' indicator button to the widget.
        # self.threshold_by_rr35._button = button
        # Flag that the widget is unset.
        # self.threshold_by_rr35._is_set = False

        # create labels to sections of RQA UI
        self.label1 = iHTML(value="<h3>Embedding Parameters</h3>", layout=self.label_layout)
        self.label2 = iHTML(value="<h3>RP Parameters</h3>", layout=self.label_layout)
        self.label3 = iHTML(value="<h3>RQA Parameters</h3>", layout=self.label_layout)
        
        # widgets to display RQA stats
        self.det = HBox([Label("Determinism:"), iHTML()])
        self.lam = HBox([Label("Laminarity:"), iHTML()])
        self.rr = HBox([Label("Recurrence Rate:"), iHTML()])
        self.tt = HBox([Label("Trapping Time:"), iHTML()])
        self.l_entr = HBox([Label("Diagonal Entropy:"), iHTML()])
        self.l_max = HBox([Label("𝐿 𝑀𝐴𝑋:"), iHTML()])
        self.l_mean = HBox([Label("𝐿 𝑀𝐸𝐴𝑁:"), iHTML()])
        self.div = HBox([Label("Divergence (1 / 𝐿 𝑀𝐴𝑋):"), iHTML()])
        self.v_entr = HBox([Label("Vertical Entropy:"), iHTML()])
        self.v_max = HBox([Label("𝑉 𝑀𝐴𝑋:"), iHTML()])
        self.v_mean = HBox([Label("𝑉 𝑀𝐸𝐴𝑁:"), iHTML()])

        # RQA inputs and output
        self.rqa_ui = VBox([self.label1, VBox([self.tau, self.dim]), Box([self.normalize]), 
            self.label2, HBox([self.metric_label, self.metric]), self.RR, 
            HBox([self.which_RR_label, self.which_RR]), Box([self.threshold]), 
            self.label3, VBox([self.theiler, self.lmin, self.vmin])],
            layout=Layout(width="500px", overflow="visible", margin_right="50px"))
        self.statistics = VBox([self.det, self.lam, self.rr, self.tt, self.v_entr, 
            self.l_max, self.l_mean, self.div, self.v_entr, self.v_max, self.v_mean],
            layout=Layout(display="flex", flex_direction="column", width="300px",
            align_self="center", justify_content="center", overflow="visible"))

        # widgets to update marks on RP figure
        self.show_rqa_lines_label = iHTML("Toggle to show:", layout=self.layout)

        self.show_theiler = ToggleButton(description='Theiler Window', value=False,
            continuous_update=False, disabled=False, style=self.style, layout=self.layout)

        self.show_lmax = ToggleButton(description='𝐿 𝑀𝐴𝑋', value=False, continuous_update=False,
            disabled=False, style=self.style, layout=self.layout)

        self.show_vmax = ToggleButton(description='𝑉 𝑀𝐴𝑋', value=False, continuous_update=False,
            disabled=False, style=self.style, layout=self.layout)

        self.show_rqa_lines = HBox([self.show_rqa_lines_label, self.show_theiler, 
            self.show_lmax, self.show_vmax], style=self.style,
            layout=Layout(width="500px", height="30px", overflow="visible", flex="1", display="flex",
            align_self="center", align_items="center", flex_direction="row", justify_content="center"))

    
    # import methods
    # from ._widgets import update_stat, update_lightcurve, update_rp, update_rp2, update_theiler, update_theiler_slider 
    # from ._widgets import update_vlines, update_longest_diag, setStyles, fix_colab_outputs, set_css
    # from ._timeseries import make_sine, make_cosine, rossler, logistic_map, disrupted_brownian, gen_normal, gen_random_walk
    # from ._utils import exclude_diagonals, transform_to_area
