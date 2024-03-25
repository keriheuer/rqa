from traitlets import dlink
import numpy as np
from ipywidgets import Layout, HBox, Box, VBox, IntSlider, FloatSlider, Dropdown, ToggleButton, ToggleButtons, Label, HTML
import pandas as pd, sys
from . import utils, timeseries
from .utils import get_resource_path
from pyunicorn.timeseries import RecurrencePlot
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GenerateRPs:
    
    def __init__(self):
        
        # parameters and variables
        self.rqa_vals = pd.read_csv(get_resource_path('data/characteristic_rqa_stats_exclude_theiler.csv'), index_col=0).to_dict(orient='index')
        self.stats = {'DET': 'Determinism', 'LAM': 'Laminarity', 'L MAX': 'Longest Diagonal Line Length', 
                        'V MAX': 'Longest Vertical Line Length', 'L MEAN': 'Average Diagonal Line Length', 
                        'TT': 'Trapping Time', 'DIV': 'Divergence (1/L MAX)', 'L ENTR': 'Diagonal Line Entropy', 'V ENTR': 'Vertical Line Entropy',
                        'C ENTR': 'Complexity Entropy', 'T REC': 'Mean Recurrence Time'}
            
        self.colors = {'White Noise': '#db57b2', 
                        'Sine': '#a157db', 
                        'Sinusoid': '#5770db', 
                        'Logistic Map': '#57d3db', 
                        'Brownian Motion': '#57db80'}
        
        self.create_widgets()
        self.setup_plots()
        self.connect_widgets()
        self.adjust_widget_layouts()
        self.display_widgets() 
        
    def create_widgets(self):
        
        # LOGISTIC MAP PARAMETERS
        self.r = utils.create_slider(3.7, 0.5, 4, 0.01, '𝑟')
        self.x = utils.create_slider(0.4, 0, 1, 0.1,'𝑥')
        self.drift = utils.create_slider(0, -0.1, 0.1, 0.01, 'Drift')

        # PERIODIC SIGNALS
        self.amplitude = utils.create_slider(1, 0.1, 5, 0.1,'𝐴')
        self.frequency = utils.create_slider(2, 1, 20, 1, '𝜔')
        self.phi = utils.create_slider(0, -3.2, 3.2, 0.2, "𝜑")
        self.N = utils.create_slider(350, 100, 500, 10, '# Steps')
        self.y0 = utils.create_slider(5, 1, 30, 1, '𝑦₀')
        self.amp1 = utils.create_slider(1, 0.1, 5, 0.1, '𝐴₁')
        self.freq1 = utils.create_slider(3, 1, 20, 1, '𝜔₁')
        self.amp2 = utils.create_slider(1, 0.1, 5, 0.1, '𝐴₂')
        self.freq2 = utils.create_slider(3, 1, 20, 1, '𝜔₂')
        self.signal1_type = Dropdown(values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])
        self.signal2_type = Dropdown(values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])
        self.mean = utils.create_slider(0, 0, 5, 0.1, '𝜇')
        self.std = utils.create_slider(1, 0.1, 5, 0.1, '𝜎')

        # ROSSLER ATTRACTOR -- BIFURCATION PARAMETERS
        self.a = utils.create_slider(0.2, -0.5, 0.55, 0.01, '𝑎')
        self.b = utils.create_slider(0.2, 0, 2, 0.01, '𝑏')
        self.c = utils.create_slider(5.7, 0, 45, 0.01, '𝑐')

        # GENERAL
        self.signal1 = VBox([self.signal1_type, self.amp1, self.freq1], layout=utils.dropdown_layout)
        self.signal2 = VBox([self.signal2_type, self.amp2, self.freq2], layout=utils.dropdown_layout)
        self.superimposed = HBox(children=[self.signal1, self.signal2], layout=utils.dropdown_layout)
        self.duration = utils.create_slider(500, 300, 1000, 10, '𝑇')
        self.duration_short = utils.create_slider(150, 100, 500, 10, '𝑇')
        self.time_step = utils.create_slider(0.01, -0.01, 0.01, 0.001, '𝑑𝑡')

        # UI widgets mapping
        self.ts_sliders = {
            'Brownian Motion': [self.y0, self.N, self.duration_short],
            'Sine': [self.frequency, self.amplitude, self.phi],
            'Cosine': [self.frequency, self.amplitude, self.phi],
            'Logistic Map': [self.r, self.x, self.duration_short, self.drift],
            'Superimposed': [self.superimposed],
            # 'Rossler': [self.a, self.b, self.c],
            'White Noise': [self.duration, self.mean, self.std]
        }
        
        self.ts_functions = {
            'Brownian Motion': timeseries.gen_random_walk(self.y0.children[0].value, n_step=self.N.children[0].value)[:self.duration.children[0].value],
            'Sine': timeseries.sine(self.frequency.children[0].value, self.amplitude.children[0].value, phi=self.phi.children[0].value),
            'Cosine': timeseries.cosine(self.frequency.children[0].value, self.amplitude.children[0].value, phi=self.phi.children[0].value),
            'Logistic Map': timeseries.logistic_map(self.r.children[0].value,self.x.children[0].value,self.drift.children[0].value,500,self.duration_short.children[0].value),
            'Superimposed': self.superimpose_signals(),
            'White Noise': timeseries.white_noise(self.duration.children[0].value, mean=self.mean.children[0].value, std=self.std.children[0].value)
        }
        
        self.signal = Dropdown(options=list(self.ts_sliders.keys()), value=list(self.ts_sliders.keys())[0], description="signal").add_class('hide-label')
        self.rqa_stat = Dropdown(options=list(self.stats.keys()), layout=utils.dropdown_layout, description="rqa metric").add_class('hide-label')
        self.tau = utils.create_slider(1, 1, 10, 1,"Time Delay (𝜏)").add_class("pad-slider")
        self.dim = utils.create_slider(1, 1, 10, 1, 'Embedding Dimension (𝑚)').add_class("pad-slider")
        self.normalize = ToggleButton(description="Normalize time series", value=True, layout=utils.button_layout).add_class('flush-left')
        self.metric = ToggleButtons(options=['Euclidean', 'Manhattan', 'Supremum'], style=utils.style, layout=utils.button_layout).add_class('no-margins')
        self.metric_label = Label('Distance Metric', layout=utils.layout).add_class("flush-left shrink-vertical-margin")
        self.RR = utils.create_slider(0.15, 0.01, 0.99, 0.01, "Recurrence Rate (𝑅𝑅)")
        self.which_RR = ToggleButtons(options=['Global 𝑅𝑅', 'Local 𝑅𝑅'], style=utils.style, layout=utils.button_layout).add_class('no-margins')
        self.threshold = ToggleButton(description='Threshold 𝑅𝑃', value=False, layout=utils.button_layout) #.add_class('flush-left')
        self.lmin = utils.create_slider(2, 2, 10, 1, '𝐿 𝑀𝐼𝑁')
        self.vmin = utils.create_slider(2, 2, 10, 1, '𝑉 𝑀𝐼𝑁')
        self.theiler = utils.create_slider(1, 1, 20, 1, "Theiler Window")
        self.threshold_by_rr35 = ToggleButton(description='Threshold by 𝑅𝑅 = 15%', value=False, style=utils.style, layout=utils.button_layout)
        
        # initialize time series user interface with white noise sliders
        self.ts_ui = VBox([self.duration, self.mean, self.std], layout=Layout(display="flex", overflow="visible"))
        self.ts_sliders = {'Brownian Motion': [self.y0, self.N, self.duration_short],
                    'Sine': [self.frequency, self.amplitude, self.phi],
                    'Cosine': [self.frequency, self.amplitude, self.phi],
                    'Logistic Map': [self.r, self.x, self.duration_short, self.drift],
                    'Superimposed': [self.superimposed],
                    'Rossler': [self.a, self.b, self.c],
                    'White Noise': [self.duration, self.mean, self.std]}

        # create labels to sections of RQA UI
        self.label1 = HTML(value="<h2>Embedding Parameters</h2>").add_class("title")
        self.label2 = HTML(value="<h2>RP Parameters</h2>").add_class("title")
        self.label3 = HTML(value="<h2>RQA Parameters</h2>").add_class("title")
        
        # RQA inputs and output
        self.rqa_ui = VBox([self.label1, self.tau, self.dim, self.normalize,  
            self.label2, HBox([self.metric_label, self.metric], style=utils.style), self.RR, 
            HBox([Label("For 𝑅𝑄𝐴, threshold by", layout=utils.layout).add_class("flush-left shrink-vertical-margin"), self.which_RR], style=utils.style),
            self.threshold, self.label3, self.theiler, self.lmin, self.vmin],
            layout=Layout(width="400px", overflow="visible", margin_right="50px"))
        
        # widgets to display RQA stats
        self.det = utils.create_text("Determinism:")
        self.lam = utils.create_text("Laminarity:")
        self.rr = utils.create_text("Recurrence Rate:")
        self.tt = utils.create_text("Trapping Time:")
        self.l_entr = utils.create_text("Diagonal Entropy:")
        self.l_max = utils.create_text("𝐿 𝑀𝐴𝑋:")
        self.l_mean = utils.create_text("𝐿 𝑀𝐸𝐴𝑁:")
        self.div = utils.create_text("Divergence (1 / 𝐿 𝑀𝐴𝑋):")
        self.v_entr = utils.create_text("Vertical Entropy:") 
        self.v_max = utils.create_text("𝑉 𝑀𝐴𝑋:")
        self.v_mean = utils.create_text("𝑉 𝑀𝐸𝐴𝑁:")

        self.statistics = utils.create_col_center(self.det, self.lam, self.rr, self.tt, self.v_entr, 
                            self.l_max, self.l_mean, self.div, self.v_entr, self.v_max, self.v_mean)
        self.statistics.layout.width = "200px"
        self.statistics.layout.padding = "initial initial initial 15px"

    def connect_widgets(self):

        dlink((self.signal, "value"), (self.ts_ui, "children"), lambda x: self.ts_sliders[x])
        
        for widget in [self.frequency, self.N, self.r, self.x, self.duration, self.duration_short, self.time_step, self.amplitude,
                self.y0, self.drift, self.phi, self.amp1, self.freq1, self.amp2, self.freq2, self.signal, self.signal1_type, self.signal2_type,
                self.mean, self.std, self.a, self.b, self.c]:
                if isinstance(widget, HBox): # a slider
                    widget.children[0].observe(self.update_lightcurve) #,names='value') #, names=['value'])
                   
                else:
                    widget.observe(self.update_lightcurve) #, names='value') #, names=['value'])
        
        self.threshold_by_rr35.observe(self.update_rp) #, names=["value"])
        
        for w in [self.tau, self.dim, self.normalize, self.metric, self.threshold, self.RR,
            self.which_RR, self.lmin, self.vmin, self.theiler, self.l_max, self.v_max]:
            if isinstance(w, HBox): #slider
                w.children[0].observe(self.update_rp2)
            else:
                w.observe(self.update_rp2) #, type="change")
                
        for w in [self.tau, self.dim, self.theiler]:
            w.children[0].observe(self.update_theiler_slider, type="change")
                    
        self.ts_plot.canvas.observe(self.update_rp, type="change")
        
    def adjust_widget_layouts(self):
        
        self.signal.layout.width = "220px"
        
        for s in [self.signal1_type, self.signal2_type]:
            s.layout.width = "200px"

        for s in [self.amp1, self.amp2, self.freq1, self.freq2]:
            s.children[0].layout.width = "230px"
             
        self.ts_plot.canvas.layout.margin = "20px 0 0 auto"
        self.threshold_by_rr35.layout.height = "40px"

        self.ts_ui.layout.margin = "0 0 0 20px"
        self.ts_plot.canvas.layout.bottom = "0"
        
        self.rp_plot.canvas.layout.margin = "15px 0 0 auto"
            
    def setup_plots(self):
        
         # create figures
        ylabel, xlabel, title, margins = "Amplitude", "Time (arb. units)", "Brownian Motion", {'x':0.02, 'y':0.02}
        self.ts_plot, self.ts_ax = utils.create_fig(6,4, xlabel=xlabel, ylabel=ylabel, title=title, position=None, margins=margins, ticks=True, layout="compressed") 
        self.rp_plot, self.rp_ax = utils.create_fig(6.5,5.5, xlabel=xlabel, ylabel=xlabel, title='Distance Matrix', margins=margins, ticks=False, layout="compressed", aspect='equal') 
        
        # initial plot
        self.ts = self.ts_functions[self.signal.value]
        self.lc, = self.ts_ax.plot(np.arange(len(self.ts)), self.ts, linewidth=1)
        self.rp = RecurrencePlot(np.array(self.ts), normalize=False, metric='euclidean', recurrence_rate=0.35, silence_level=2)   
        self.rp_matrix = self.rp_ax.matshow(self.rp._distance_matrix, origin='lower', cmap='plasma', interpolation='none')
        self.rp_cb = self.add_colorbar(self.rp_plot, self.rp_ax, self.rp_matrix, cbar_title=r'$\epsilon$')

    def display_widgets(self):
        
        # render the figure alongside UI
        left = VBox([HBox([self.signal, self.ts_ui], layout=Layout(display="flex", flex_direction="row", height="100px", left="0")).add_class("left-row"), 
                     self.ts_plot.canvas], layout=Layout(display="flex", flex_direction="column")).add_class("left")
       
        right = VBox([self.threshold_by_rr35, self.rp_plot.canvas],
                        layout=Layout(display="flex", flex_direction="column", justify_items="center", align_items="center", right="0")).add_class("right")
        row = HBox([left, right],
                    layout=Layout(width="1300px", display="flex", flex_direction="row", justify_items="center", align_items='stretch')).add_class("row")
        display(row)
        
        display(HTML(""" <style> 
        .hide-label > .widget-label {
           display: none !important
        }
        </style>"""))

    def get_superimposed_title(self):
        
        ts1_label = r"$A_1 sin(\omega_1 t)$" if self.signal1_type.value == 'Sine' else r"$A_1 cos(\omega_1 t)$"
        ts2_label = r"$A_2 sin(\omega_2 t)$" if self.signal2_type.value == 'Sine' else r"$A_2 cos(\omega_2 t)$"
        return r"$y =$ %s + %s" % (ts1_label, ts2_label)

    def superimpose_signals(self):
  
        y1 = timeseries.sine(self.freq1.children[0].value, self.amp1.children[0].value) if self.signal1_type == 'Sine' else timeseries.cosine(self.freq1.children[0].value, self.amp1.children[0].value)
        y2 = timeseries.sine(self.freq2.children[0].value, self.amp2.children[0].value) if self.signal2_type.value == 'Sine' else timeseries.cosine(self.freq2.children[0].value, self.amp2.children[0].value)
        return y1 * y2

    def update_lightcurve(self, change):
        """ Displays time series based on signal type and slider values."""

        titles = {'Brownian Motion': 'Brownian Motion',
                  'Sine': r"$y = A sin(\omega t) + \phi$",
                  'Cosine': r"$y = A cos(\omega t) + \phi$",
                  'Logistic Map': 'Logistic Map',
                  'Superimposed': self.get_superimposed_title(),
                  'White Noise': 'White Noise'
                  }
        
        ts_functions = {
        'Brownian Motion': timeseries.gen_random_walk(self.y0.children[0].value, n_step=self.N.children[0].value)[:self.duration.children[0].value], 
            'Sine': timeseries.sine(self.frequency.children[0].value, self.amplitude.children[0].value, phi=self.phi.children[0].value),
            'Cosine': timeseries.cosine(self.frequency.children[0].value, self.amplitude.children[0].value, phi=self.phi.children[0].value),
            'Logistic Map': timeseries.logistic_map(self.r.children[0].value,self.x.children[0].value,self.drift.children[0].value,500,self.duration_short.children[0].value),
            'Superimposed': self.superimpose_signals(),
            'White Noise': timeseries.white_noise(self.duration.children[0].value, mean=self.mean.children[0].value, std=self.std.children[0].value)
        }
        
        self.ts_ax.set_title(titles[self.signal.value], fontsize=16) 
        self.ts = ts_functions[self.signal.value]
        self.lc.set_ydata(self.ts)
        self.lc.set_xdata(np.arange(len(self.ts)))
        self.ts_ax.relim()
        self.ts_ax.autoscale(enable=True, axis="both")
        
        # re-draw artists and flush events
        self.ts_plot.canvas.draw()
        self.ts_plot.canvas.flush_events()

    def update_rp(self, *args):
        """ Updates RP in generate_rp() based on
        current time series."""
        
        # retrieve current light curve
        ts = self.ts_ax.lines[0].get_ydata()

        # update RP instance
        if self.which_RR.value.split()[0] == "Global":
            self.rp = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), recurrence_rate=self.RR.children[0].value, tau=self.tau.children[0].value, dim=self.dim.children[0].value, silence_level=2)
        else:
            self.rp = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), local_recurrence_rate=self.RR.children[0].value, tau=self.tau.children[0].value, dim=self.dim.children[0].value, silence_level=2)

        # update RP properties
        if self.threshold_by_rr35.value:
            data, cmap, title, show_cb = self.rp.recurrence_matrix(), 'Greys', 'Recurrence Plot', False
        else:
            data, cmap, title, show_cb = self.rp._distance_matrix, 'plasma', "Distance Matrix", True
            
        new_clims = [np.min(data), np.max(data)]
        self.rp_matrix.set(array=data, cmap=cmap, clim=new_clims)
        self.rp_ax.set_title(title, fontsize=16)
        self.rp_cb.ax.set_visible(show_cb)
        self.refresh_rp(self.rp_plot, self.rp_ax, self.rp_matrix, self.rp_cb, data)
          
    def refresh_rp(self, fig, ax, rp_matrix, cb, data):
        
        # update extent and tick_params
        utils.update_extent(rp_matrix, data)
        ax.tick_params(which="both", axis="both", right=False, bottom=False, top=False, left=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
        cb.ax.tick_params(which="both", axis="both", right=True, bottom=False, top=False, left=False, labelright=True, direction='out')
        cb.ax.tick_params(which="minor", direction='out', length=3)
        cb.ax.tick_params(which="major", direction='out', length=8)

        # re-draw artists and flush events
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
    def varyParams(self):
        
        ylabel, xlabel, title, margins = "Amplitude", "Time (arb. units)", self.signal.value, {'x':0.02, 'y':0.02}
        self.rp_plot2, self.rp_ax2 = utils.create_fig(6.5,5.5, xlabel=xlabel, ylabel=xlabel, title='Distance Matrix', margins=margins, ticks=False, layout="compressed", aspect='equal') 
        self.rp_matrix2 = self.rp_ax2.matshow(self.rp._distance_matrix, origin='lower', cmap='plasma', interpolation='none')
        self.rp_cb2 = self.add_colorbar(self.rp_plot2, self.rp_ax2, self.rp_matrix2, cbar_title=r'$\epsilon$')
        self.rp_ax2.tick_params(which="both", axis="both", right=False, bottom=False, top=False, left=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
        utils.update_extent(self.rp_matrix2, self.rp._distance_matrix)
        
        # retrieve current light curve
        ts = self.ts_ax.lines[0].get_ydata()

        # update RP instance
        if self.which_RR.value.split()[0] == "Global":
            self.rp2 = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), recurrence_rate=self.RR.children[0].value, tau=self.tau.children[0].value, dim=self.dim.children[0].value, silence_level=2)
        else:
            self.rp2 = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), local_recurrence_rate=self.RR.children[0].value, tau=self.tau.children[0].value, dim=self.dim.children[0].value, silence_level=2)
            
        self.update_rqa_vals(self.rp, self.lmin.children[0].value, self.vmin.children[0].value)
        
        # horizontally display RQA UI, RP figure, and RQA stats
        display(utils.create_row(self.rqa_ui, self.rp_plot2.canvas, self.statistics))
       
        self.ts_plot.canvas.observe(self.update_rp2, type="change")
                
        self.rp_plot2.canvas.layout.margin = "auto 0 0 auto"
           
    def update_rp2(self, *args):
        """ Updates RP in generate_rp() based on
        current time series."""
        
        # retrieve current light curve
        ts = self.ts_ax.lines[0].get_ydata()

        # update RP instance
        if self.which_RR.value.split()[0] == "Global":
            self.rp2 = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), recurrence_rate=self.RR.children[0].value, tau=self.tau.children[0].value, dim=self.dim.children[0].value, silence_level=2)
        else:
            self.rp2 = RecurrencePlot(ts, normalize=self.normalize.value, metric=self.metric.value.lower(), local_recurrence_rate=self.RR.children[0].value, tau=self.tau.children[0].value, dim=self.dim.children[0].value, silence_level=2)

        # update RP properties
        if self.threshold.value:
            data, cmap, title, show_cb = self.rp2.recurrence_matrix(), 'Greys', 'Recurrence Plot', False
        else:
            data, cmap, title, show_cb = self.rp2._distance_matrix, 'plasma', "Distance Matrix", True
            
        new_clims = [np.min(data), np.max(data)]
        self.rp_matrix2.set(array=data, cmap=cmap, clim=new_clims)
        self.rp_ax2.set_title(title, fontsize=16)
        self.rp_cb2.ax.set_visible(show_cb)
            
        new_clims = [np.min(data), np.max(data)]
        self.rp_matrix.set(array=data, cmap=cmap, clim=new_clims)
        self.rp_ax.set_title(title, fontsize=16)
        self.rp_cb.ax.set_visible(show_cb)
        self.refresh_rp(self.rp_plot2, self.rp_ax2, self.rp_matrix2, self.rp_cb2, data)
        
        # update RQA
        l_min, v_min = self.lmin.children[0].value, self.vmin.children[0].value
        self.update_rqa_vals(self.rp2, l_min, v_min)
       
    def update_rqa_vals(self, rp, l_min, v_min):
        
        self.rr.children[1].value = "%.2f" % rp.recurrence_rate()
        
         # get vertical based measures first
        self.lam.children[1].value = "%.2f" % rp.laminarity(v_min=v_min)
        self.v_entr.children[1].value = "%.2f" % rp.vert_entropy(v_min=v_min)
        self.tt.children[1].value = "%.2f" % rp.trapping_time(v_min=v_min)
        self.v_max.children[1].value = "%d" % rp.max_vertlength()
        self.v_mean.children[1].value = "%d" % rp.average_vertlength(v_min=v_min)

        # exclude diagonals
        self.exclude_diagonals()

        # then get diagonal based measures
        self.det.children[1].value = "%.2f" % rp.determinism(l_min=l_min)
        self.l_max.children[1].value = "%d" % rp.max_diaglength()
        self.l_mean.children[1].value = "%d" % rp.average_diaglength(l_min=l_min)
        self.div.children[1].value = "%.2f" % (1/rp.max_diaglength())
        self.l_entr.children[1].value = "%.2f" % rp.diag_entropy(l_min=l_min)

    def update_theiler_slider(self, *args):
        if self.tau.children[0].value != 1 and self.dim.children[0].value != 1:
            self.theiler.children[0].max = (self.tau.children[0].value * self.dim.children[0].value) + 10

    def exclude_diagonals(self):
        
        # clear pyunicorn cache for RQA
        self.rp2.clear_cache(irreversible=True)

        # set width of Theiler window (n=1 for main diagonal only)
        n = self.theiler.children[0].value

        # create mask to access corresponding entries of recurrence matrix
        mask = np.zeros_like(self.rp2.R, dtype=bool)

        for i in range(len(self.rp2.R)):
            mask[i:i+n, i:i+n] = True

        # set diagonal (and subdiagonals) of recurrence matrix to zero
        # by directly accessing the recurrence matrix rp.R
        self.rp2.R[mask] = 0
        
    def add_colorbar(self, fig, ax, sc, cbar_title="", **kwargs):
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.08)
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.set_title(cbar_title, fontsize='xx-large')
        cb.ax.tick_params(which="minor", direction='out', length=3)
        cb.ax.tick_params(which="major", direction='out', length=8, pad=5)
        cb.ax.margins(y=0.02)
        cb.ax.set_autoscale_on(True)
        cb.outline.set_linewidth(1.25)
        return cb
    
