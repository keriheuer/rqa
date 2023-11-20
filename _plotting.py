""" Methods for creating various figures
used by the RQA widget and notebook."""

# display set up
from IPython.display import display, Javascript, HTML, Math, Markdown
import numpy as np
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


from ._utils import find_longest_diag, find_vmax, find_lmax, transform_to_area, exclude_diagonals, validate


class WidgetPlots():
    def __init__(self):
        self.label = 'WidgetPlots'

    def generate_rps(self):

        #  initialize UI with sine wave
        # with self.ui:
            # display(VBox([self.frequency], layout=Layout(display="flex", overflow="visible")))

        # initialize time series figure with sine
        self.ts = self.white_noise(duration=self.duration.value, mean=self.mean.value, std=self.std.value)

        # scales and axes for time series figure
        self.x_scale, self.y_scale = LinearScale(), LinearScale()
        self.xax = Axis(label="Time (arb. units)",scale=self.x_scale)
        self.yax = Axis(label="Amplitude",scale=self.y_scale,orientation="vertical")

        # create lines mark
        self.line = Lines(x=np.arange(len(self.ts)), y=self.ts, scales={"x": self.x_scale, "y": self.y_scale}, stroke_width=1)

        # create time series figure
        self.ts_plot = Figure(marks=[self.line], axes=[self.xax, self.yax], title='White Noise',
        layout=Layout(height="400px", width="500px", display="flex", overflow="visible"), padding_y=0, style=self.style)
        self.ts_plot.layout.margin_right = "50px"

        # turn off bqplot grid
        plt.grids(value="none")

        # scales and axes for RP figure
        self.cs = ColorScale(scheme="plasma")
        self.y_scale2 = LinearScale()

        # create RP instance
        rp = RecurrencePlot(np.array(self.ts), normalize=False, metric='euclidean', recurrence_rate=0.35, silence_level=2)
        self.domain = np.arange(len(rp._distance_matrix))
        
        # create HeatMap mark
        self.rp_matrix = HeatMap(x=self.domain, y=self.domain, color=rp._distance_matrix, scales={'color': self.cs, "x": self.x_scale, "y": self.y_scale2})
        
        # additional axis and scale for RP
        self.cax = ColorAxis(scale=self.cs, label=r"𝜀", orientation="vertical", side="right", grid_color="black", max=np.max(self.domain))
        self.yax2 = Axis(label="Time (arb. units)",scale=self.x_scale,orientation="vertical")

        # create RP figure
        self.rp_plot = Figure(
            marks=[self.rp_matrix],
            axes=[self.xax, self.yax2, self.cax],
            title="Distance Matrix",
            layout=Layout(width="500px", height="500px", overflow="visible"),
           # fig_margin = dict(top=100, bottom=100, left=0, right=100),
            min_aspect_ratio=1, max_aspect_ratio=1,
            padding_y=0,
            style=self.style
        )

        
        # turn off bqplot grid
        plt.grids(value="none")

        # register callbacks by using the 'observe' method
        self.signal.observe(self.update_ts_ui, names=['value'])
        self.signal.observe(self.update_lightcurve, names=['value'])
        
        for w in [self.frequency, self.N, self.r, self.x, self.duration, self.duration_short, self.time_step, self.amplitude,
                self.y0, self.drift, self.amp1, self.freq1, self.amp2, self.freq2, self.signal1_type, self.signal2_type,
                self.mean, self.std, self.a, self.b, self.c]:
                w.observe(self.update_lightcurve, names=['value'])

        self.line.observe(self.update_rp)

        # set the interval selector on the time series figure
        # self.window_selector = BrushIntervalSelector(marks=[self.line], scale=self.line.scales["x"])
        # self.ts_plot.interaction = self.window_selector

        # register callback with brushing trait of interval selector
        # self.window_selector.observe(self.update_rp, names=["brushing"])
        # self.line.observe(self.update_rp, names=["brushing"])

        self.rp_plot.observe(self.update_rp)
        self.rp_matrix.observe(self.update_rp)
        self.threshold_by_rr35.layout.height = "40px"
        self.threshold_by_rr35.observe(self.update_rp, names=["value"])
        self.threshold_by_rr35.observe(self.update_rp_colorbar, names=["value"])
      
    
        # render the figure alongside UI
        display(HBox([VBox([VBox([self.signal, self.ts_ui]), self.ts_plot]), VBox([self.threshold_by_rr35, self.rp_plot], layout=Layout(display="flex", flex_flow="vertical", justify_items="center", align_items="center"))]))

        # update widget instance properties
        self.R = rp.recurrence_matrix()
        self.distances = rp._distance_matrix
     

    def vary_rqa_params(self):

        # create scales
        self.cs2 = ColorScale(scheme="plasma")
        self.x_sc = LinearScale(allow_padding=False)
        self.y_sc = LinearScale(allow_padding=False)

        # create axes
        self.c_ax = ColorAxis(scale=self.cs2, label=r"𝜀", orientation="vertical", side="right", grid_color="black", max=np.max(self.domain))
        self.x_ax = Axis(label="Time (arb. units)", scale=self.x_sc)
        self.y_ax = Axis(label="Amplitude", scale=self.y_sc,orientation="vertical")

        # use RP from generate_rp() method
        if self.validate:
            self.line.observe(self.update_rp2, type="change")
            domain = self.domain # np.arange(len(fig2.marks[0].color))
            dist = self.distances #fig2.marks[0].color
        else:
            # generate random signal based on Dropdown value
            # pass
            print("ERROR: must run previous cell first")

        # create marks
        self.rp_matrix2 = HeatMap(color=dist, x=domain, y=domain, scales={"x": self.x_sc, "y": self.y_sc, 'color': self.cs2})
        self.theiler_line = Lines(x=domain, y=[domain, domain], scales={'x': self.x_sc, 'y': self.y_sc},
            fill_colors='red', fill="inside", fill_opacities=[0, 0], stroke_width=0, apply_clip=True, preserve_domain={'x': True, 'y':True})
        self.vlines = Lines(x=domain, y=domain, scales={"x": self.x_sc, "y": self.y_sc}, stroke_width=1, colors=['green'], opacities=[0])
        self.longest_diag = Lines(x=domain, y=domain, scales={"x": self.x_sc, "y": self.y_sc}, stroke_width=1, colors=['blue'], opacities=[0])

        # create Figure
        self.rp2_plot = Figure(
            marks=[self.rp_matrix2, self.theiler_line, self.vlines, self.longest_diag],
            axes=[self.c_ax, self.x_ax, self.y_ax],
            title="Distance Matrix",
            layout=Layout(width="500px", height="500px", overflow="visible", margin="0px 50px"),
            min_aspect_ratio=1, max_aspect_ratio=1,
            padding_y=0, padding_x=0,
            style=self.style)
        # fig3.layout.margin="0px 50px"

        # turn off bqplot grid
        plt.grids(value="none")

        # display RQA
        rqa_stats = Output()
        with rqa_stats:
            display(self.statistics) #, layout=Layout(display="flex", flex_direction="column", align_self="center"))

        # register callbacks
        for w in [self.tau, self.dim, self.normalize, self.metric, self.threshold, self.RR,
            self.which_RR, self.lmin, self.vmin, self.theiler, self.l_max, self.v_max]:
            w.observe(self.update_rp2, type="change")

        for w in [self.tau, self.dim, self.theiler]:
            w.observe(self.update_theiler_slider, type="change")
            w.observe(self.update_theiler, names=["value"])

        # for w in [self.l_max, self.threshold, self.show_lmax]:
        #     w.observe(self.update_longest_diag, type="change")

        # for w in [self.v_max, self.threshold, self.show_vmax]:
        #     w.observe(self.update_vlines, type="change")

       # self.threshold.observe(self.update_rp_colorbar, names=["value"])
      
        # horizontally display RQA UI, RP figure, and RQA stats
        display(HBox([self.rqa_ui, VBox([self.rp2_plot, self.show_rqa_lines]), rqa_stats])) #, layout=Layout(display="flex", overflow="visible",
            # align_items="center", flex_direction="column", justify_items="center")), rqa_stats],
            # layout=Layout(display="flex", flex_direction="row", justify_content="space-between",
            # align_items="center", overflow="visible")))

        # update widget instance properties
        # self.rp2_plot = fig3
        # self.rp_matrix2 = rp_matrix2
        # self.theiler_line = theiler_line
        # self.vlines = vlines
        # self.longest_diag = longest_diag

    def characteristic_rps(self):

        # make sure plots are aligned to center of output
        display(HTML("""
        <style>
        #output-body {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """))

        # load in time series data
        white_noise = np.load(".data/example_timeseries/white_noise_ts.npy")
        sine = np.load(".data/example_timeseries/sine_ts.npy")
        super_sine = np.load(".data/example_timeseries/super_sine_ts.npy")
        logi = np.load(".data/example_timeseries/logistic_map_ts.npy")
        brownian = np.load(".data/example_timeseries/brownian_ts.npy")

        # create RP instances
        white_noise_rp = RecurrencePlot(white_noise, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=2, dim=2)
        sine_rp = RecurrencePlot(sine, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=18, dim=6)
        super_sine_rp = RecurrencePlot(super_sine, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05,tau=8, dim=5)
        logi_rp = RecurrencePlot(logi, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=5, dim=10)
        brownian_rp = RecurrencePlot(brownian, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=2, dim=2)

        # create figure and arrays to loop through
        fig, ax = pplt.subplots(nrows=2, ncols=5, figsize=(20, 8), gridspec_kw={"height_ratios": [1,1]})
        signals = [white_noise_rp, sine_rp, super_sine_rp, logi_rp, brownian_rp]
        timeseries = [white_noise, sine, super_sine, logi, brownian]
        titles = ['White Noise', 'Sine', 'Superimposed Sine', 'Logistic Map', 'Brownian Motion']

        for i, (ts, rp) in enumerate(zip(timeseries, signals)):

            c = self.colors[i]

            # for clarity, truncate x-axis of plot so signal variability is easily seen
            if i == 2 or i == 4:
                ax[0, i].plot(ts[:300], linewidth=1, color=c)
            elif i == 3:
                ax[0,i].plot(ts[:50], linewidth=1, color=c)
            elif i == 1:
                ax[0,i].plot(ts[:500], linewidth=1, color=c)
            elif i == 0:
                ax[0,i].plot(ts[:400], linewidth=1, color=c)

            # plot with either matshow() or imshow() with no interpolation between points
            ax[1,i].imshow(rp.recurrence_matrix(), cmap='Greys', origin='lower', interpolation='none')

            # turn off axis ticks and set title
            ax[0,i].tick_params(which="both", axis="both", bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False, labeltop=False)
            ax[0,i].set_title(titles[i])
            ax[1,i].tick_params(which="both", axis="both", bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False, labeltop=False)
            

        # render figure
        fig.subplots_adjust(wspace=0.05, hspace=0.02)
        pplt.show()

    def characteristic_rqa(self):

        # load in RQA stats
        df = pd.read_csv('.data/characteristic_systems_rqa_exclude_theiler.csv')

        # create plotly figure widget to show bar graph of RQA
        layout = {'yaxis': {'range': [0, df['DET'].max()*1.1]}}
        self.rqa_bar_plot = go.FigureWidget(data=[go.Bar(
            x=['White Noise', 'Sine', 'Superimposed Sine','Logistic Map', 'Brownian Motion'],
            y=df['DET'].values,
            marker_color=self.colors,
            text=df['DET'].values,
            textposition="outside",
            texttemplate="%{y:.2%}",
            hoverinfo='skip')])

        # update plot sizing
        self.rqa_bar_plot.update_layout(
            # width=700,
            width=1500,
            height=400,
            autosize=False,
            margin=dict(t=50, b=0, l=0, r=0),
            template="plotly_white",
            hovermode=False,
            yaxis_title='Fraction of Recurrences',
            title_x=0.5)

        self.bar = self.rqa_bar_plot.data[0]

        # register callbacks
        self.rqa_bar_plot.observe(self.update_stat, type="change")
        self.rqa_stat.observe(self.update_stat, type="change")

        # render Dropdown and bar plot vertically
        display(VBox([self.rqa_stat, self.rqa_bar_plot]))

    # UTILIY FUNCS
    # def find_longest_diag(self, array2D, patternlength):
    #     res=[]
    #     for k in range(-array2D.shape[0]+1, array2D.shape[1]):
    #         diag=np.diag(array2D, k=k)
    #         if(len(diag)>=patternlength):
    #             for i in range(len(diag)-patternlength+1):
    #                 if(all(diag[i:i+patternlength]==1)):
    #                     res.append((i+abs(k), i) if k<0 else (i, i+abs(k)))
    #     return res

    # def find_vmax(self, distances, i, vmax):
    # # find start positions of all vertical lines with length = V MAXß
    #     col = distances[i, :]
    #     idx_pairs = np.where(np.diff(np.hstack(([False],col==1,[False]))))[0].reshape(-1,2)
    #     lengths = np.diff(idx_pairs,axis=1) # island lengths
    #     if vmax in lengths:
    #         return (i, idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]) # longest island start position


    # def find_lmax(self, array2D, lmax):
    # # find start position of diagonal with length = L MAX
    #     res = []
    #     for k in range(-array2D.shape[0]+1, array2D.shape[1]):
    #         diag = np.diag(array2D, k=k)
    #         if(len(diag) >= lmax):
    #             for i in range(len(diag) - lmax + 1):
    #                 if(all(diag[i:i + lmax] == 1)):
    #                     res.append((i + abs(k), i) if k<0 else (i, i + abs(k)))
    #     return res

    # def transform_to_area(self, x, y1, y2):

    #     """ Helper function for filling between lines in bqplot figure."""

    #     return np.append(x, x[::-1]), np.append(y1, y2[::-1])

    # def exclude_diagonals(self, rp, theiler):

    #     # copy recurrence matrix for plotting
    #     rp_matrix = rp.recurrence_matrix()

    #     # clear pyunicorn cache for RQA
    #     rp.clear_cache(irreversible=True)

    #     # set width of Theiler window (n=1 for main diagonal only)
    #     n = theiler

    #     # create mask to access corresponding entries of recurrence matrix
    #     mask = np.zeros_like(rp.R, dtype=bool)

    #     for i in range(len(rp_matrix)):
    #         mask[i:i+n, i:i+n] = True

    #     # set diagonal (and subdiagonals) of recurrence matrix to zero
    #     # by directly accessing the recurrence matrix rp.R

    #     rp.R[mask] = 0

    # # check if cell generating fig2 was run, i.e. rp_plot is not empty
    # def validate(self):
    #     if self.rp_plot != None:
    #         return True
    #     else:
    #         return False
