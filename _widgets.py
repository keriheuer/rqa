""" Methods for styling ipywidgets output
and updating widgets / plots in RQA notebook."""

# display set up

from IPython.display import display, Javascript, HTML, Math, Markdown
import numpy as np, sys, os, subprocess
# widgets
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, Layout, HBox, Box, VBox, Output
from ipywidgets import IntSlider, FloatSlider, Dropdown, ToggleButton, ToggleButtons, Label
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

# suppress numpy warning for elementwise comparison
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# plotting
try:
    import seaborn as sns, pandas as pd
except:
    # use dummy warnings
    np.linalg._umath_linalg._ilp64 = getattr(np, '_ilp64', dummy_npwarn_decorator_factory)
    np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)
    import seaborn as sns, pandas as pd

import plotly.graph_objects as go, plotly.express as px
import matplotlib as mpl, matplotlib.pyplot as pplt
from matplotlib_inline.backend_inline import set_matplotlib_formats

def resize_colab_cell():
  display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 7000})'))

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: nowrap;
    }

    .bqplot > .svg-background {
      width: 500px !important;
      height: 500px !important;
      }

    .bqplot > svg .brushintsel .selection {
      fill: gold;
    }

    .modebar {
      display: none !important;
    }

    .classic {
    --bq-axis-tick-text-fill: black;
    --bq-axis-border-color: black;
   # --bq-axis-tick-stroke: black;
    --bq-font: "serif";
    --bq-axis-tick-text-font: "18px serif";
    --bq-axislabel-font: 18px "serif";
    --bq-mainheading-font: 18px "serif"; 
    --bq-axis-path-stroke: black;}

   .bqplot .figure .jupyter-widgets .classic {
    font-family: "serif"
    }
               
    #output-body {
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 100%;
        max-width: 100%;
    }

    .output_png {
        display: flex;
        align-items: center;
        justify-content: center;
    }

  </style>
  '''))

def fix_colab_outputs():
    cmd = "pip install --upgrade watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)

    output_style = """
        <style>
           .jupyter-widgets-output-area .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
            .jupyter-widgets-output-area  {
                height: auto !important;
            }
        </style>
        """

    display(HTML(output_style))

def setStyles():
    get_ipython().events.register('pre_run_cell', set_css)
    set_matplotlib_formats('svg')
    # Colab setup ------------------
    if "google.colab" in sys.modules:
        
        get_ipython().events.register('pre_run_cell', resize_colab_cell)
        display(Javascript("google.colab.output.resizeIframeToContent()"))
        display(Javascript('''google.colab.output.setIframeWidth(0, true, {maxWidth: 500})'''))
        fix_colab_outputs()

        # register custom widgets (bqplot)
        from google.colab import output
        output.enable_custom_widget_manager()


class WidgetBase():

    def __init__(self):
        self.label = 'WidgetBase'
        setStyles()

    def update_lightcurve(self, *args):
        """ Displays time series based on signal type and slider values."""
        
        if self.signal.value == 'Brownian Motion':
            self.ts = self.gen_random_walk(self.y0.value, n_step=self.N.value)[:self.duration_short.value]
            self.ts_plot.title = "Brownian Motion"

        elif self.signal.value == 'Sine':
            self.ts = self.sine(self.frequency.value, self.amplitude.value, phi=self.phi.value)
            self.ts_plot.title = "𝑦 = 𝐴𝑠𝑖𝑛(𝜔𝑡) + 𝜑"

        elif self.signal.value == 'Cosine':
            self.ts = self.cosine(self.frequency.value, self.amplitude.value)
            self.ts_plot.title = "𝑦 = 𝐴𝑐𝑜𝑠(𝜔𝑡) + 𝜑"

        elif self.signal.value == "Logistic Map":
            self.ts = self.logistic_map(self.r.value,self.x.value,self.drift.value,500,self.duration_short.value)
            self.ts = self.ts + self.drift.value*np.arange((self.duration_short.value))
            self.ts_plot.title = "Logistic Map"

        elif self.signal.value == 'Superimposed':
            if self.signal1_type.value == 'Sine':
                ts1 = self.sine(self.freq1.value, self.amp1.value)
                ts1_label = "𝐴₁𝑠𝑖𝑛(𝜔₁𝑡)"
            else:
                ts1 = self.cosine(self.freq1.value, self.amp1.value)
                ts1_label = "𝐴₁𝑐𝑜𝑠(𝜔₁𝑡)"

            if self.signal2_type.value == 'Sine':
                ts2 = self.sine(self.freq2.value, self.amp2.value)
                ts2_label = "𝐴₂𝑠𝑖𝑛(𝜔₂𝑡)"
            else:
                ts2 = self.cosine(self.freq2.value, self.amp2.value)
                ts2_label = "𝐴₂𝑐𝑜𝑠(𝜔₂𝑡)"

            self.ts_plot.title = "𝑦 = %s + %s" % (ts1_label, ts2_label)
            self.ts = ts1 * ts2

        elif self.signal.value == 'Rossler':
            self.ts = self.rossler(self.a.value,self.b.value,self.c.value,0,10,0.01)
            self.ts_plot.title = "Rossler Attractor"

        elif self.signal.value == 'White Noise':
            self.ts = self.white_noise(self.duration.value, mean=self.mean.value, std=self.std.value)
            self.ts_plot.title = "White Noise"

        with self.line.hold_sync():
            self.line.y = self.ts
            self.line.x = np.arange(len(self.ts))

    def update_ts_ui(self, *args):
        """ Displays controls for desired time series."""

        if self.signal.value == 'Brownian Motion':
            # sliders = VBox([N,M], layout=Layout(display="flex", overflow="visible"))
            self.ts_ui.children = [self.y0, self.N, self.duration_short]

        elif self.signal.value == 'Sine':
            # sliders = VBox([frequency, amplitude], layout=Layout(display="flex", overflow="visible"))
            self.ts_ui.children = [self.frequency, self.amplitude, self.phi]

        elif self.signal.value == 'Cosine':
            # sliders = VBox([frequency, amplitude], layout=Layout(display="flex", overflow="visible"))
            self.ts_ui.children = [self.frequency, self.amplitude, self.phi]

        elif self.signal.value == "Logistic Map":
            # sliders = VBox([r,x,M], layout=Layout(display="flex", overflow="visible"))
            self.ts_ui.children = [self.r, self.x, self.duration_short, self.drift]

        elif self.signal.value == "Superimposed":
            # sliders = VBox([r,x,M], layout=Layout(display="flex", overflow="visible"))
            self.ts_ui.children = [self.superimposed]

        elif self.signal.value == 'Rossler':
            self.ts_ui.children = [self.a, self.b, self.c]

        elif self.signal.value == 'White Noise':
            self.ts_ui.children = [self.duration, self.mean, self.std]

    def update_rp(self, *args):
        """ Updates RP in generate_rp() based on
        current time series."""


        # retrieve current time series
        ts = self.line.y

        # brushing = self.window_selector.brushing
        # if not brushing:
        #     # interval selector is active
        #     if self.line.selected is not None:
        #         # get the start and end indices of the interval
        #         start_ix, end_ix = self.line.selected[0], self.line.selected[-1]
        #     # interval selector is *not* active
        #     else:  
        #         start_ix, end_ix = 0, -1

        # update RP instance
        if self.which_RR.value.split()[0] == "Global":
            rp = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), recurrence_rate=self.RR.value, tau=self.tau.value, dim=self.dim.value, silence_level=2)
        else:
            rp = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), local_recurrence_rate=self.RR.value, tau=self.tau.value, dim=self.dim.value, silence_level=2)

        self.distances = rp._distance_matrix
        self.R = rp.recurrence_matrix()
        self.rp_instance = rp
        self.domain = np.arange(len(self.R))

        # update RP heatmap
        # with self.rp_matrix.hold_sync():
        # print(self.threshold_by_rr35.value)
        # with self.rp_plot.hold_sync():
            # if not brushing:
            #     self.rp_matrix.x, self.rp_matrix.y = np.arange(len(self.distances))[start_ix:end_ix], np.arange(len(self.distances))[start_ix:end_ix]
            #     self.rp_matrix.color = self.distances[start_ix:end_ix]
            # else:

        with self.rp_plot.hold_sync():
            if self.threshold_by_rr35.value == True:
                self.rp_matrix.color = rp.recurrence_matrix()
                self.cs.scheme = "Greys"
                self.rp_plot.title = "Recurrence Plot"
                self.cax.visible = False
            else:   
                self.rp_matrix.color = rp._distance_matrix
                self.cs.scheme = "plasma"
                self.rp_plot.title = "Distance Matrix"
                self.cax.visible = True
            self.rp_matrix.x, self.rp_matrix.y = self.domain, self.domain

    def update_rp_colorbar(self, *args):
        if self.threshold_by_rr35.value == True:
          self.cax.visible = False
        else:
          self.cax.visible = True

    def update_rp2_colorbar(self, *args):
        if self.threshold_by_rr35.value == True:
          self.c_ax.visible = False
        else:
          self.c_ax.visible = True
      
    def update_rp2(self, *args):
        """ Updates RP from generate_rp() based on
        current value of RQA params."""

        # update RP instance
        if self.which_RR.value.split()[0] == "Global":
            rp = RecurrencePlot(self.ts, normalize=self.normalize.value, metric=self.metric.value.lower(), recurrence_rate=self.RR.value, tau=self.tau.value, dim=self.dim.value, silence_level=2)
        else:
            rp = RecurrencePlot(self.ts, normalize=self.normalize.value, metric=self.metric.value.lower(), local_recurrence_rate=self.RR.value, tau=self.tau.value, dim=self.dim.value, silence_level=2)

        self.distances = rp._distance_matrix
        self.R = rp.recurrence_matrix()
        self.rp_instance = rp
        self.domain = np.arange(len(self.R))

        # plot RP before excluding diagonals
        with self.rp2_plot.hold_sync():

            # threshold the RP by recurrence rate
            if self.threshold.value == True:
                self.rp2_plot.title = "Recurrence Plot"
                self.cs2.scheme = "Greys"
                self.rp_matrix2.color = rp.recurrence_matrix()
                self.cax.visible = False

            # plot unthresholded recurrence plot (i.e. the distance matrix)
            else:
                self.rp2_plot.title = "Distance Matrix"
                self.cs2.scheme = "plasma"
                self.c_ax.max = np.max(rp._distance_matrix)
                self.rp_matrix2.color = rp._distance_matrix
                self.cax.visible = True

        
        # update RQA
        l_min, v_min = self.lmin.value, self.vmin.value

        self.rr.children[1].value = "%.2f" % rp.recurrence_rate()
        self.lam.children[1].value = "%.2f" % rp.laminarity(v_min=v_min)
        self.v_entr.children[1].value = "%.2f" % rp.vert_entropy(v_min=v_min)
        self.tt.children[1].value = "%.2f" % rp.trapping_time(v_min=v_min)
        self.v_max.children[1].value = "%d" % rp.max_vertlength()
        self.v_mean.children[1].value = "%d" % rp.average_vertlength(v_min=v_min)

        # exclude diagonals
        mask = self.exclude_diagonals(rp, self.theiler.value)

        self.det.children[1].value = "%.2f" % rp.determinism(l_min=l_min)
        self.l_max.children[1].value = "%d" % rp.max_diaglength()
        self.l_mean.children[1].value = "%d" % rp.average_diaglength(l_min=l_min)
        self.div.children[1].value = "%d" % (1/rp.max_diaglength())
        self.l_entr.children[1].value = "%.2f" % rp.diag_entropy(l_min=l_min)

        # update_rqa_lines(rp.recurrence_matrix())

    def update_theiler(self, *args):

        # distances = self.rp_matrix2.color

        if self.threshold.value:
            with self.theiler_line.hold_sync():
                y1, y2 = np.arange(len(self.distances))-self.theiler.value, np.arange(len(self.distances))+self.theiler.value
                x_area, y_area = transform_to_area(np.arange(len(self.distances)), y1, y2)
                y_area[y_area < 0] = 0
                y_area[y_area > len(self.distances)] = len(self.distances)
                self.theiler_line.x = x_area
                self.theiler_line.y = y_area

                if self.show_theiler.value:
                    self.theiler_line.fill_opacities = [0.3]*2
                else:
                    self.theiler_line.fill_opacities = [0]*2

    def update_theiler_slider(self, *args):
        if self.tau.value != 1 and self.dim.value != 1:
            self.theiler.max = (self.tau.value * self.dim.value) + 10
      
    # def update_vlines(self, *args):

    # # distances = self.rp_matrix2.color

    #     if self.threshold.value:

    #         # find positions
    #         # vertical line positions
    #         length = int(self.v_max.children[1].value)
    #         # return lists of arrays with vertical lines x and y positions
    #         idx = np.array([self.find_vmax(self.distances, i, length) for i in range(len(self.distances))], dtype=object)
    #         idx = idx[idx != None]
    #         v_x, v_y = [], []
    #         for (x,y) in idx:
    #             v_x.append(np.array([x]*length))
    #             v_y.append(np.arange(y, y+length))

    #         # update vertical lines mark
    #         with self.vlines.hold_sync():
    #             self.vlines.x = v_x
    #             self.vlines.y = v_y
    #             self.vlines.colors = ['green']*len(idx)

    #         if self.show_vmax.value:
    #             self.vlines.opacities = [1]*len(idx)
    #         else:
    #             self.vlines.opacities = [0]*len(idx)

    # def update_longest_diag(self, *args):

    # # distances = self.rp_matrix2.color

    #     if self.threshold.value:

    #         # diagonal line position
    #         length = int(self.l_max.children[1].value)
    #         loc = self.find_lmax(self.distances, length)[0]
    #         l_x, l_y = np.arange(loc[0],loc[0]+length), np.arange(loc[1],loc[1]+length)

    #         # update diagonal lines mark
    #         with self.longest_diag.hold_sync():
    #             self.longest_diag.x = l_x
    #             self.longest_diag.y = l_y
    #             self.longest_diag.colors = ['blue']

    #         if self.show_lmax.value:
    #             self.longest_diag.opacities = [1]
    #         else:
    #             self.longest_diag.opacities = [0]

    def update_stat(self, *args):
        """ Helper function to update statistics in characteristic RQA plot.
        """

        # load in RQA stats
        df = pd.read_csv('rqa/data/characteristic_systems_rqa_exclude_theiler.csv')

        labels = np.array(['Determinism', 'Laminarity', 'Longest Diagonal Line Length', 'Average Diagonal Line Length', 'Longest Vertical Line Length', 'Average Vertical Line Length', 'Diagonal Line Entropy', 'Vertical Line Entropy', 'Divergence (1/L MAX)', 'Trapping Time'])
        stats = np.array(['DET', 'LAM', 'L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'L ENTR', 'V ENTR', 'DIV', 'TT'])
        # stats_formatted = np.array(['𝐷𝐸𝑇', '𝐿𝐴𝑀', '𝐿 𝑀𝐴𝑋', '𝐿 𝑀𝐸𝐴𝑁', '𝑉 𝑀𝐴𝑋', '𝑉 𝑀𝐸𝐴𝑁', '𝐿 𝐸𝑁𝑇𝑅', '𝑉 𝐸𝑁𝑇𝑅', '𝐷𝐼𝑉', '𝑇𝑇'])

        # update bars with current stat
        i = np.where(self.rqa_stat.value == stats)[0][0]
        vals = df[stats[i]].values
        self.bar.y = vals
        self.bar.text = vals

        if self.rqa_stat.value in ['L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'TT']: #['𝐿 𝑀𝐴𝑋', '𝐿 𝑀𝐸𝐴𝑁', '𝑉 𝑀𝐴𝑋', '𝑉 𝑀𝐸𝐴𝑁']: 
            self.bar.texttemplate="%{y:d}" # integer
            ylabel = 'Time (arb. units)'
        elif self.rqa_stat.value == 'DIV': #'𝐷𝐼𝑉':
            self.bar.texttemplate="%{y:.2f}" # integer
            ylabel = '1 / Time (arb. units)'
        elif 'ENTR' in self.rqa_stat.value: #'𝐸𝑁𝑇𝑅'
            self.bar.texttemplate="%{y:.2f}" # 2 decimal float, unitless
            ylabel = ""
        else:
            self.bar.texttemplate="%{y:.2%}" # 2 decimal percentage
            ylabel = 'Fraction of Recurrences'

        # adjust y axis range if needed and title
        self.rqa_bar_plot.for_each_yaxis(lambda x: x.update({'range': [0, df[stats[i]].max() * 1.1]}))
        self.rqa_bar_plot.update_layout(title_text=labels[i], yaxis_title=ylabel)
