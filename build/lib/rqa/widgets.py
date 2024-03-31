
from . import utils, timeseries
from .config import *

class GenerateRPs:
    
    def __init__(self):
        
        # LOGISTIC MAP PARAMETERS
        self.r = utils.create_slider(3, 2, 4, 0.01, '𝑟')
        self.x = utils.create_slider(0.617, 0, 1, 0.001,'𝑥')
        self.drift = utils.create_slider(0, -0.3, 0.3, 0.01, 'Drift')

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
        self.signal1_type = Dropdown(description='Signal 1',values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])
        self.signal2_type = Dropdown(description='Signal 2',values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])
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
        self.colors = {'White Noise': '#db57b2', 
                        'Sine': '#a157db', 
                        'Sinusoid': '#5770db', 
                        'Logistic Map': '#57d3db', 
                        'Brownian Motion': '#57db80'}
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
        
        self.signal = Dropdown(options=list(self.ts_sliders.keys()), value=list(self.ts_sliders.keys())[0])
  
        self.rqa_vals = pd.read_csv('rqa/data/characteristic_rqa_stats_exclude_theiler.csv', index_col=0).to_dict(orient='index')
        self.stats = {'DET': 'Determinism', 'LAM': 'Laminarity', 'L MAX': 'Longest Diagonal Line Length', 
                        'V MAX': 'Longest Vertical Line Length', 'L MEAN': 'Average Diagonal Line Length', 
                        'TT': 'Trapping Time', 'DIV': 'Divergence (1/L MAX)', 'L ENTR': 'Diagonal Line Entropy', 'V ENTR': 'Vertical Line Entropy',
                        'C ENTR': 'Complexity Entropy', 'T REC': 'Mean Recurrence Time'}
            
        self.rqa_stat = Dropdown(options=list(self.stats.keys()), layout=utils.dropdown_layout)
        self.tau = utils.create_slider(1, 1, 10, 1,"Time Delay (𝜏)").add_class("pad-slider")
        self.dim = utils.create_slider(1, 1, 10, 1, 'Embedding Dimension (𝑚)').add_class("pad-slider")
        self.normalize = ToggleButton(description="Normalize time series", value=True, layout=utils.button_layout).add_class('flush-left')
        self.metric = ToggleButtons(options=['Euclidean', 'Manhattan', 'Supremum'], style=utils.style, layout=utils.button_layout).add_class('no-margins')
        self.metric_label = Label('Distance Metric', layout=utils.layout).add_class("flush-left shrink-vertical-margin")
        self.RR = utils.create_slider(0.35, 0.01, 0.99, 0.01, "Recurrence Rate (𝑅𝑅)")
        self.which_RR = ToggleButtons(options=['Global 𝑅𝑅', 'Local 𝑅𝑅'], style=utils.style, layout=utils.button_layout).add_class('no-margins')
        self.threshold = ToggleButton(description='Threshold 𝑅𝑃', value=False, layout=utils.button_layout) #.add_class('flush-left')
        self.lmin = utils.create_slider(2, 2, 10, 1, '𝐿 𝑀𝐼𝑁')
        self.vmin = utils.create_slider(2, 2, 10, 1, '𝑉 𝑀𝐼𝑁')
        self.theiler = utils.create_slider(1, 1, 20, 1, "Theiler Window")
        self.threshold_by_rr35 = ToggleButton(description='Threshold by 𝑅𝑅 = 35%', value=False, style=utils.style, layout=utils.button_layout)
        
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
            layout=Layout(width="500px", overflow="visible", margin_right="50px"))
        
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

    def connect_widgets(self):

        dlink((self.signal, "value"), (self.ts_ui, "children"), lambda x: self.ts_sliders[x])
        
        for widget in [self.frequency, self.N, self.r, self.x, self.duration, self.duration_short, self.time_step, self.amplitude,
                self.y0, self.drift, self.phi, self.amp1, self.freq1, self.amp2, self.freq2, self.signal, self.signal1_type, self.signal2_type,
                self.mean, self.std, self.a, self.b, self.c]:
                if isinstance(widget, HBox): # a slider
                    widget.children[0].observe(self.update_lightcurve, names=['value'])
                else:
                    widget.observe(self.update_lightcurve, names=['value'])
        
        self.threshold_by_rr35.observe(self.update_rp, names=["value"])
        
        self.ts_plot.canvas.observe(self.update_rp, type="change")

        for w in [self.tau, self.dim, self.normalize, self.metric, self.threshold, self.RR,
            self.which_RR, self.lmin, self.vmin, self.theiler, self.l_max, self.v_max]:
            if isinstance(w, HBox): #slider
                w.children[0].observe(self.update_rp2)
            else:
                w.observe(self.update_rp2, type="change")
                
        for w in [self.tau, self.dim, self.theiler]:
            w.children[0].observe(self.update_theiler_slider, type="change")
            
        self.ts_plot.canvas.observe(self.update_rp2, type="change")
        
    def adjust_widget_layouts(self):
        
        self.signal.layout.width = "220px"
        
        for s in [self.signal1_type, self.signal2_type]:
            s.layout.width = "200px"

        for s in [self.amp1, self.amp2, self.freq1, self.freq2]:
            s.children[0].layout.width = "230px"
             
        self.ts_plot.canvas.layout.margin_right = "50px"
        self.threshold_by_rr35.layout.height = "40px"

        for widget in [self.ts_ui, self.ts_plot.canvas]: widget.layout.margin_right = "200px"
        self.ts_plot.canvas.layout.bottom = "0"
            
    def setup_plots(self):
        
         # create time series figure
        ylabel, xlabel, title, margins = "Amplitude", "Time (arb. units)", "White Noise", {'x':0.02, 'y':0.02}
        self.ts_plot, self.ts_ax = utils.create_fig(6,4, xlabel=xlabel, ylabel=ylabel, title=title, position=None, margins=margins, ticks=True, layout="compressed") 

        # create RP figure
        self.rp_plot, self.rp_ax = utils.create_fig(6.5,5.5, xlabel=xlabel, ylabel=xlabel, title='Distance Matrix', margins=margins, ticks=False, layout="compressed", aspect='equal') 
        
    def init_plot(self):
        
        self.lc, = self.ts_ax.plot(np.arange(len(self.ts)), self.ts)
        self.rp = RecurrencePlot(np.array(self.ts), normalize=False, metric='euclidean', recurrence_rate=0.35, silence_level=2)   
        self.rp_matrix = self.rp_ax.matshow(self.rp._distance_matrix, origin='lower', cmap='plasma', interpolation='none')
        self.rp_cb = utils.add_colorbar(self.rp_plot, self.rp_ax, self.rp_matrix, cbar_title=r'$\epsilon$')
        

    def display_widgets(self):
        
        # render the figure alongside UI
        left = VBox([HBox([self.signal, self.ts_ui], layout=Layout(display="flex", flex_direction="row", margin_right="250px", height="100px", left="0")).add_class("left-row"), 
                     self.ts_plot.canvas], layout=Layout(display="flex", flex_direction="column")).add_class("left")
       
        right = VBox([self.threshold_by_rr35, self.rp_plot.canvas],
                        layout=Layout(display="flex", flex_direction="column", justify_items="center", align_items="center", margin_left="200px", right="0")).add_class("right")
        row = HBox([left, right],
                    layout=Layout(width="1300px", display="flex", flex_direction="row", justify_items="center", align_items='stretch')).add_class("row")
                                    # justify_content="space-betselfeen", , align_self="center")).add_class("roself")
        display(row)
    
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
        
        if change['type'] == 'change' and change['name'] == 'value':
            self.ts = self.ts_functions[change["new"]]
            self.ts_ax.set_title(titles[change["new"]])
            self.lc.set_ydata(self.ts)
            self.lc.set_xdata(np.arange(len(self.ts)))
            self.ts_ax.relim(visible_only=True)
            # self.ts_ax.autoscale(enable=True, axis="both")
            self.ts_ax.autoscale_view()

            # re-draw artists and flush events
            self.ts_plot.canvas.draw()
            self.ts_plot.canvas.flush_events()
      
    def update_rp(self, *args):
        """ Updates RP in generate_rp() based on
        current time series."""
        
        # retrieve current light curve
        ts = self.ts_ax.lines[0].get_ydata()
        # ts = self.lc.get_ydata()

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
        self.rp_ax.set_title(title)
        self.rp_cb.ax.set_visible(show_cb)

        # re-draw artists and flush events
        self.update_extent(self.rp_matrix, data)
        self.rp_plot.canvas.draw_idle()
        self.rp_plot.canvas.flush_events()
        
    def varyParams(self):
        
        ylabel, xlabel, title, margins = "Amplitude", "Time (arb. units)", "White Noise", {'x':0.02, 'y':0.02}
        self.rp_plot2, self.rp_ax2 = utils.create_fig(6.5,5.5, xlabel=xlabel, ylabel=xlabel, title='Distance Matrix', margins=margins, ticks=False, layout="compressed", aspect='equal') 
        self.rp_matrix2 = self.rp_ax2.matshow(self.rp._distance_matrix, origin='lower', cmap='plasma', interpolation='none')
        self.rp_cb2 = utils.add_colorbar(self.rp_plot, self.rp_ax, self.rp_matrix, cbar_title=r'$\epsilon$')
        self.rp_ax2.tick_params(which="both", axis="both", right=False, bottom=False, top=False, left=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
        self.update_extent(self.rp_matrix2, self.rp._distance_matrix)
        

        self.update_rqa_vals(self.rp, self.lmin.children[0].value, self.vmin.children[0].value)
        
        # horizontally display RQA UI, RP figure, and RQA stats
        display(utils.create_row(self.rqa_ui, self.rp_plot2.canvas, self.statistics))
       
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
        if self.threshold_by_rr35.value:
            data, cmap, title, show_cb = self.rp2.recurrence_matrix(), 'Greys', 'Recurrence Plot', False
        else:
            data, cmap, title, show_cb = self.rp2._distance_matrix, 'plasma', "Distance Matrix", True
            
        new_clims = [np.min(data), np.max(data)]
        self.rp_matrix2.set(array=data, cmap=cmap, clim=new_clims)
        self.rp_ax2.set_title(title)
        self.rp_cb2.ax.set_visible(show_cb)

        # re-draw artists and flush events
        self.update_extent(self.rp_matrix2, data)
        self.rp_plot2.canvas.draw_idle()
        self.rp_plot2.canvas.flush_events()
        
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
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(sc, cax=cax, label=cbar_title)
        return cb

class CombineTimeseries():

    def __init__(self):
        
        # WidgetBase.__init__(self)
        # TimeSeries.__init__(self)
        # Callbacks.__init__(self)
        self.cmap = sns.color_palette("hls", 8).as_hex()[::-1][:5]
        self.systems = np.array(['White Noise', 'Sine', 'Sinusoid', 'Logistic Map', 'Brownian Motion'])
        self.stats = np.array(['DET', 'LAM', 'L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'L ENTR', 'V ENTR', 'DIV', 'TT'])
        
        # load pre-generated time series data
        self.white_noise = np.load("rqa/data/example_timeseries/white_noise_ts.npy")
        self.sine = np.load("rqa/data/example_timeseries/sine_ts.npy")
        self.super_sine = np.load("rqa/data/example_timeseries/super_sine_ts.npy")
        self.logi = np.load("rqa/data/example_timeseries/logistic_map_ts.npy")
        self.brownian = np.load("rqa/data/example_timeseries/brownian_ts.npy")
        self.signals_ts = [self.white_noise, self.sine, self.super_sine, self.logi, self.brownian]
        
        self.colors, self.timeseries = {}, {}
        self.rqa_vals = {}
        ts_data = [self.white_noise, self.sine, self.super_sine, self.logi, self.brownian]
        df = pd.read_csv('rqa/data/characteristic_systems_rqa_exclude_theiler.csv')
        for i, s in enumerate(self.systems):
            self.colors[s] = self.cmap[i]
            self.timeseries[s] = ts_data[i]
            self.rqa_vals[s] = {}
            for stat, val in zip(df.columns, df.iloc[i].values):
                self.rqa_vals[s][stat] = val
                
        self.ts1_type = Dropdown(options=self.systems, value='Brownian Motion', layout=Layout(width="200px"))
        self.ts2_type = Dropdown(options=self.systems, value='Sine', layout=Layout(width="200px"))
        self.ts3_type = Dropdown(options=self.systems, value='White Noise', layout=Layout(width="200px"))
        
        self.ts1_amp = HBox([FloatSlider(value=1, min=0.05, max=2, step=0.01, readout=False, continuous_update=False, description='𝐴₁', layout=Layout(width="150px", height="25px", overflow="visible")),
                             Label("1", layout=Layout(width="fit-content"))])
        self.ts2_amp = HBox([FloatSlider(value=1, min=0.05, max=2, step=0.01, readout=False, continuous_update=False, description='𝐴₂', layout=Layout(width="150px", height="25px", overflow="visible")),
                             Label("1", layout=Layout(width="fit-content"))])
        self.ts3_amp = HBox([FloatSlider(value=1, min=0.05, max=2, step=0.01, readout=False, continuous_update=False, description='𝐴\u2083', layout=Layout(width="150px", height="25px", overflow="visible")),
                             Label("1", layout=Layout(width="fit-content"))])
        
        self.ts1_duration = HBox([IntSlider(value=300, min=200, max=1000, step=10, readout=False, continuous_update=False, description='𝑇₁', layout=Layout(width="150px", height="25px", overflow="visible")), Label("200", layout=Layout(width="fit-content"))])
        self.ts2_duration = HBox([IntSlider(value=300, min=200, max=1000, step=10, readout=False, continuous_update=False, description='𝑇₂', layout=Layout(width="150px", height="25px", overflow="visible")), Label("200", layout=Layout(width="fit-content"))])
        self.ts3_duration = HBox([IntSlider(value=300, min=200, max=1000, step=10, readout=False, continuous_update=False, description='𝑇\u2083', layout=Layout(width="150px", height="25px", overflow="visible")), Label("200", layout=Layout(width="fit-content"))])

        self.ts1 = VBox([self.ts1_type, self.ts1_amp, self.ts1_duration], layout=Layout(display="flex", flex_direction="column", align_items="center", width="33.33%", justify_content="space-between"))
        self.ts2 = VBox([self.ts2_type, self.ts2_amp, self.ts2_duration], layout=Layout(display="flex", flex_direction="column", align_items="center", width="33.33%", justify_content="space-between"))
        self.ts3 = VBox([self.ts3_type, self.ts3_amp, self.ts3_duration], layout=Layout(display="flex", flex_direction="column", align_items="center", width="33.33%", justify_content="space-between"))

        self.ts_list = HBox([self.ts1, self.ts2, self.ts3], layout=Layout(display="flex", flex_direction="row", height="110px", top="0%", align_self="flex-end", width="90%"))
        self.ts_list.layout.margin = "0 0 15px 0"
        
        self.spliced_rr_slider = FloatSlider(min=0, max=0.25, value=0.1, step=0.01, continuous_update=False, readout=False, layout=Layout(width="150px", overflow="visible"))
        self.spliced_rr_label = Label("0.10", layout=Layout(width="fit-content"))
        self.spliced_rr = HBox([Label('𝑅𝑅'), self.spliced_rr_slider, self.spliced_rr_label])
        
        self.show_splices = ToggleButton(value=False, description="Outline RP windows", layout=Layout(width="auto"))

        self.spliced_tau_slider = IntSlider(min=1, max=20, value=1, step=1, continuous_update=False, description="𝜏  ", readout=False, layout=Layout(width="150px", overflow="visible"))
        self.spliced_tau_label = Label("1", layout=Layout(width="fit-content"))
        self.spliced_tau = HBox([self.spliced_tau_slider, self.spliced_tau_label])
        
  
        self.spliced_dim_slider =  IntSlider(min=1, max=8, value=1, step=1, continuous_update=False, description="𝑚  ", readout=False, layout=Layout(width="150px", overflow="visible"))
        self.spliced_dim_label = Label("1", layout=Layout(width="fit-content"))
        self.spliced_dim = HBox([self.spliced_dim_slider, self.spliced_dim_label])
        
        self.spliced_rp_ui_left = VBox([self.spliced_rr, self.show_splices])
        self.spliced_rp_ui_left.layout.margin = "0 0 0 25px"
        self.spliced_rp_ui_right = VBox([self.spliced_tau, self.spliced_dim])
        self.spliced_rp_ui_right.layout.margin = "0 10px 0 0"
        
        self.spliced_rp_ui = HBox([self.spliced_rp_ui_left, self.spliced_rp_ui_right], layout=Layout(display="flex", flex_direction="row", justify_content="space-between", height="60px", right="0", align_self="stretch"))
        self.spliced_rp_ui.layout.margin = "0 0 15px 0"
        
        self.spliced_rqa_stat = Dropdown(options=['DET', 'LAM', 'RR', 'L MAX', 'L MEAN', 'L ENTR', 'DIV', 'V MAX', 'TT',  'V ENTR'], value='DET',
                                            layout=Layout(width="150px", align_self="flex-end", bottom="0"))
        
        self.sig1 = HTML(value='<p style="font-size: 14px"><font color="%s">' % self.colors[self.ts1_type.value] + '%s' % self.ts1_type.value + '</font></p>')
        self.sig2 = HTML(value='<p style="font-size: 14px"><font color="%s">' % self.colors[self.ts2_type.value] + '%s' % self.ts2_type.value + '</font></p>')
        self.sig3 = HTML(value='<p style="font-size: 14px"><font color="%s">' % self.colors[self.ts3_type.value] + '%s' % self.ts3_type.value + '</font></p>')
        
        self.sig1_rqa = HTML("", style=style, continuous_update=False)
        self.sig2_rqa = HTML("", style=style, continuous_update=False)
        self.sig3_rqa = HTML("", style=style, continuous_update=False)
        
        self.full_spliced = HTML('<p style="font-size: 14px">Combined</p>', style=style)
        self.full_spliced_rqa = HTML("", style=style, continuous_update=False)

        self.all_rqa = HBox([self.spliced_rqa_stat,
                            VBox([self.sig1, self.sig1_rqa], layout=col),
                            VBox([self.sig2, self.sig2_rqa], layout=col),
                            VBox([self.sig3, self.sig3_rqa], layout=col),
                            VBox([self.full_spliced, self.full_spliced_rqa], 
                            layout=col)], layout=row)
        self.all_rqa.layout.margin = "12px 0 0 0"
        
        self.connect_widgets()
        self.setup_plots()
        
    def connect_widgets(self):
        
        # link sliders and labels
        for slider in [self.ts1_amp, self.ts2_amp, self.ts3_amp]:
            dlink((slider.children[0], "value"), (slider.children[1], "value"), lambda x: "%.2f" % x)
            slider.children[0].observe(self.update_spliced_rp, type="change")

        for slider in [self.ts1_duration, self.ts2_duration, self.ts3_duration]:
            dlink((slider.children[0], "value"), (slider.children[1], "value"), lambda x: "%d" % x)
            slider.children[0].observe(self.update_durations, type="change")

        dlink((self.spliced_rr_slider, "value"), (self.spliced_rr_label, "value"), lambda x: "%.2f" % x) # self.format_slider_label(x))
       
        dlink((self.spliced_tau_slider, "value"), (self.spliced_tau_label, "value"), lambda x: "%d" % x) # self.format_slider_label(x))
        
        dlink((self.spliced_dim_slider, "value"), (self.spliced_dim_label, "value"), lambda x: "%d" % x) # self.format_slider_label(x))
    
        # widget callbacks
        for widget in [self.ts1_type, self.ts1_duration.children[0], self.ts1_amp.children[0]]:
            widget.observe(self.update_splice1, type="change")
            widget.observe(self.update_spliced_rp, type="change")

        for widget in [self.ts2_type, self.ts2_duration.children[0], self.ts2_amp.children[0]]:
            widget.observe(self.update_splice2, type="change")
            widget.observe(self.update_spliced_rp, type="change")

        for widget in [self.ts3_type, self.ts3_duration.children[0], self.ts3_amp.children[0]]:
            widget.observe(self.update_splice3, type="change")
            widget.observe(self.update_spliced_rp, type="change")

        self.show_splices.observe(self.outline_splices, type="change")

        for callback in [self.update_spliced_rqa, self.update_spliced_rp, self.update_splice1, self.update_splice2, self.update_splice3]:
            for widget in [self.spliced_tau_slider, self.spliced_dim_slider, self.spliced_rr_slider, self.spliced_rqa_stat]:
                widget.observe(callback, type="change")
        
        display(HTML("<style>.container { width:100% !important; } div.output_subarea {padding: 2em 0 0;} .jupyter-matplotlib {height:fit-content; width:fit-content} .widget-html > .widget-html-content {line-height: 20px; height: 20px;} .jupyter-widgets label {line-height: 25px; height: 25px; min-width: 16px !important; width: fit-content !important; margin-right: 2px !important} .widget-hslider .slider-container, .jupyter-widget-hslider .slider-container {margin-left: 10px !important} </style>"))
    
    def setup_plots(self):

        # light curve
        x, y = 'Time (arb. units)', 'Amplitude'
        margins = {'x': 0.002, 'y': 0.05}
        position = {'bottom': 0.15, 'top': 0.99, 'left': 0.1, 'right': 0.85}
        self.combined_ts, self.combined_ts_ax = self.create_default_fig(7, 4, xlabel=x, ylabel=y, margins=margins, layout="constrained", ticks=True) #position=position,
    
        # normalize the data to same scale
        spliced = [self.brownian, self.sine, self.white_noise]
        T = 300
        spliced = [normalize(s[:T], n) for s,n in zip(spliced, ['Brownian Motion', 'Sine', 'White Noise'])]

        # connect time series
        offset = spliced[0][-1] - spliced[1][0]
        spliced[1] = spliced[1] + offset
        offset2 = spliced[1][-1] - spliced[2][0]
        spliced[2] = spliced[2] + offset2
    
        # combine into one light curve
        spliced_ts = self.splice_ts(spliced)

        # plot light curves
        self.splice1, = self.combined_ts_ax.plot(np.arange(T), spliced[0], self.colors['Brownian Motion'], linewidth=1)
        self.splice2, = self.combined_ts_ax.plot(np.arange(T, 2*T), spliced[1], self.colors['Sine'], linewidth=1)
        self.splice3, = self.combined_ts_ax.plot(np.arange(2*T, 3*T), spliced[2], self.colors['White Noise'], linewidth=1)
        
        # RP
        x, y = 'Time (arb. units)', 'Time (arb. units)'
        position = {'bottom': 0.08, 'top': 0.99, 'left': 0.2, 'right': 0.85}
        self.combined_rp, self.combined_rp_ax = self.create_default_fig(4.5, 4.5, xlabel=x, ylabel=y, layout="constrained", ticks=False) #position=position,
            
        rp_instance, rp_matrix = get_rp_matrix(spliced_ts, self.spliced_rr_slider.value, tau=self.spliced_tau_slider.value, dim=self.spliced_dim_slider.value)
        self.combined_rp_matrix = plot_rp_matrix(self.combined_rp_ax, rp_matrix)
        update_extent(self.combined_rp_matrix, rp_matrix)
        self.full_spliced_rqa.value =  get_rqa_stat(rp_instance, self.spliced_rqa_stat.value, l_min=2, v_min=2)
        
        # RQA
        self.update_spliced_rqa()
                           
        # draw RP window outlines
        self.window1 = self.combined_rp_ax.add_patch(Rectangle((2, 2), len(spliced[0])-2, len(spliced[0])-2, alpha=1, fill='none', facecolor='none', edgecolor=self.colors['Brownian Motion'], linewidth=2))
        self.window2 = self.combined_rp_ax.add_patch(Rectangle((len(spliced[0]), len(spliced[0])), len(spliced[1]), len(spliced[1]), alpha=1, fill='none', facecolor='none', edgecolor=self.colors['Sine'], linewidth=2))
        self.window3 = self.combined_rp_ax.add_patch(Rectangle((len(spliced[0])+len(spliced[1]), len(spliced[0])+len(spliced[1])), len(spliced[2])-2, len(spliced[2])-2, alpha=1, fill='none',facecolor='none', edgecolor=self.colors['White Noise'], linewidth=2))

        # hide RP window outlines initially
        for w in [self.window1, self.window2, self.window3]: w.set_visible(False)

        # render widgets
        left_half = VBox([self.ts_list, self.combined_ts.canvas], layout=Layout(display="flex", flex_direction="column", align_items="flex-end", left="0", justify_content="space-between", justify_items="flex-end"))
        right_half = VBox([self.spliced_rp_ui, self.combined_rp.canvas], layout=Layout(display="flex", flex_direction="column", align_items="flex-end", right="0", justify_items="flex-end"))
        spliced_ui = HBox([left_half, right_half], layout=Layout(display="flex", flex_direction="column", align_items="stretch", justify_content="space-between"))
        display(VBox([spliced_ui, self.all_rqa], layout=Layout(display="flex", flex_direction="column", align_items="stretch"))) 

    def update_splice1(self, *args):

        """ Update the first time series in the light
        curve plot and its sub-RP color. Shift second
        and third time series if duration changed. """
        
        # get new data
        T1 = self.ts1_duration.children[0].value
        new_ts = self.timeseries[self.ts1_type.value][:T1]
        new_ts = normalize(new_ts, self.ts1_type.value) * self.ts1_amp.children[0].value
        self.splice1.set_ydata(new_ts)
        self.splice1.set_xdata(np.arange(0, T1))
        self.splice1.set_color(self.colors[self.ts1_type.value])
        
        # update limits and flush fig
        self.redraw_ts()
        self.connect_splices()
        
        # udpate RP window color
        self.window1.set(edgecolor=self.colors[self.ts1_type.value])
        self.redraw_rp()
        
        # update RQA val
        rp_instance, rp_matrix = get_rp_matrix(new_ts, self.spliced_rr_slider.value, tau=self.spliced_tau_slider.value, dim=self.spliced_dim_slider.value)
        rqa_val = get_rqa_stat(rp_instance, self.spliced_rqa_stat.value, theiler=self.spliced_tau_slider.value*self.spliced_dim_slider.value, l_min=2, v_min=2)
        self.sig1_rqa.value = '<p style="font-size: 14px">%s</p>' % rqa_val
        
    def update_splice2(self, *args):

        """ Update the second time series in the light
        curve plot and its sub-RP color. """
            
        # get new data
        T1 = self.ts1_duration.children[0].value
        T2 = self.ts2_duration.children[0].value
        new_ts2 = self.timeseries[self.ts2_type.value][:T2]
        new_ts2 = normalize(new_ts2, self.ts2_type.value) * self.ts2_amp.children[0].value

        offset = self.splice1.get_ydata()[0] - self.splice2.get_ydata()[-1]
        self.splice2.set_ydata(new_ts2 + offset)
        self.splice2.set_xdata(np.arange(T1, T1+T2))
        self.splice2.set_color(self.colors[self.ts2_type.value])
        
        # update limits and flush fig
        self.redraw_ts()
        self.connect_splices()
        
        # udpate RP window color
        self.window2.set(edgecolor=self.colors[self.ts2_type.value])
        self.redraw_rp()
        
        # update RQA val
        rp_instance, rp_matrix = get_rp_matrix(self.splice2.get_ydata(), self.spliced_rr_slider.value, tau=self.spliced_tau_slider.value, dim=self.spliced_dim_slider.value)
        rqa_val = get_rqa_stat(rp_instance, self.spliced_rqa_stat.value, theiler=self.spliced_tau_slider.value*self.spliced_dim_slider.value, l_min=2, v_min=2)
        self.sig2_rqa.value = '<p style="font-size: 14px">%s</p>' % rqa_val
        

    def update_splice3(self, *args):

        """ Update the last time series in the light
        curve plot and its sub-RP color. """
        
        # get new data
        T1 = self.ts1_duration.children[0].value
        T2 = self.ts2_duration.children[0].value
        T3 = self.ts3_duration.children[0].value
        new_ts = self.timeseries[self.ts3_type.value][:T3]
        new_ts = normalize(new_ts, self.ts3_type.value) * self.ts3_amp.children[0].value
        
        offset = self.splice2.get_ydata()[-1] - new_ts[0]
        self.splice3.set_ydata(new_ts + offset)
        self.splice3.set_xdata(np.arange(T1+T2, T1+T2+T3))
        self.splice3.set_color(self.colors[self.ts3_type.value])
        
        # update limits and flush fig
        self.redraw_ts()
        self.connect_splices()
        
        # update RP window color
        self.window3.set(edgecolor=self.colors[self.ts3_type.value])
        self.redraw_rp()
        
        # update RQA
        rp_instance, rp_matrix = get_rp_matrix(self.splice3.get_ydata(), self.spliced_rr_slider.value, tau=self.spliced_tau_slider.value, dim=self.spliced_dim_slider.value)
        rqa_val = get_rqa_stat(rp_instance, self.spliced_rqa_stat.value, theiler=self.spliced_tau_slider.value*self.spliced_dim_slider.value, l_min=2, v_min=2)
        self.sig3_rqa.value = '<p style="font-size: 14px">%s</p>' % rqa_val
        
    def redraw_ts(self, idle=True):
                
        self.combined_ts_ax.relim()
        self.combined_ts_ax.autoscale_view(True,True,True)
        
        if idle:
            self.combined_ts.canvas.draw_idle()
        else:
            self.combined_ts.canvas.draw()
        self.combined_ts.canvas.flush_events()
        
    def redraw_rp(self, idle=True):
        if idle: 
            self.combined_rp.canvas.draw_idle()
        else: 
            self.combined_rp.canvas.flush_events()
        
    def connect_splices(self):

        x1, y1 = self.splice1.get_xdata(), self.splice1.get_ydata()
        x2, y2 = self.splice2.get_xdata(), self.splice2.get_ydata()

        T1 = self.ts1_duration.children[0].value
        T2 = self.ts2_duration.children[0].value
        T3 = self.ts3_duration.children[0].value
        
        # check if ts1 and ts2 are continuous
        matched = (x1[-1] == x2[0]) and (y1[-1] == y2[0])
        
        if not matched:
            self.splice2.set_xdata(np.arange(T1, T1+T2)) # align ts2 to end of ts1
            offset = y1[-1] - y2[0] # vertical offset
            self.splice2.set_ydata(y2 + offset)
            self.redraw_ts()
            
        # make sure to get the *current* ts2 after matching
        x2, y2 = self.splice2.get_xdata(), self.splice2.get_ydata()
        x3, y3 = self.splice3.get_xdata(), self.splice3.get_ydata()
    
       # check if ts2 and ts3 are continuous
        matched = (x2[-1] == x3[0]) and (y2[-1] == y3[0])
        
        if not matched:
            self.splice3.set_xdata(np.arange(T1+T2, T1+T2+T3)) # align ts3 to end of ts2
            offset = y2[-1] - y3[0] # vertical offset
            self.splice3.set_ydata(y3 + offset)
            self.redraw_ts()
        
#         # normalize full spliced ts after making continuous
#         spliced_ts = self.splice_ts([self.splice1.get_ydata(), self.splice2.get_ydata(), self.splice3.get_ydata()])
#         normed_ts = (spliced_ts - np.mean(spliced_ts)) / np.std(spliced_ts)
        
#         # update each splice using T1, T2, T3 
#         self.splice1.set_ydata(normed_ts[:T1])
#         self.splice2.set_ydata(normed_ts[T1:T1+T2])
#         self.splice3.set_ydata(normed_ts[T1+T2:])
#         self.redraw_ts()
        
    def update_durations(self, *args):
        
        x1 = self.splice1.get_xdata()
        x2 = self.splice2.get_xdata()
        x3 = self.splice3.get_xdata()
        
        # get duration of each splice
        T1 = self.ts1_duration.children[0].value
        T2 = self.ts2_duration.children[0].value
        T3 = self.ts3_duration.children[0].value

        # update RP windows
        self.window1.set(width=T1-2, height=T1-2)
        self.window2.set(width=T2, height=T2, xy=(T1, T1))
        self.window3.set(width=T3, height=T3, xy=(T1+T2-2, T1+T2-2))
        self.redraw_rp()

    def update_spliced_rp(self, *args):
        """ Plots the RP for the current spliced time series. """

        spliced_ts = self.splice_ts([self.splice1.get_ydata(), self.splice2.get_ydata(), self.splice3.get_ydata()])

        rp = RecurrencePlot(spliced_ts, metric="euclidean", recurrence_rate=self.spliced_rr_slider.value, silence_level=2, tau=self.spliced_tau_slider.value, dim=self.spliced_dim_slider.value)
        self.combined_rp_matrix.set_data(rp.recurrence_matrix())
        self.combined_rp_ax.tick_params(which="both", axis="both", bottom=False, top=False, left=False, right=False, labeltop=False, labelbottom=True, labelleft=True)
        update_extent(self.combined_rp_matrix, rp.recurrence_matrix()) # update extent just in case
        self.redraw_rp()
        
        # update spliced_ts RQA stat after plotting
        self.full_spliced_rqa.value = get_rqa_stat(rp, self.spliced_rqa_stat.value, theiler=self.spliced_tau_slider.value*self.spliced_dim_slider.value, l_min=2, v_min=2)

    def update_spliced_rqa(self, *args):
        """ Update RQA stats to be for the selected metric via the `spliced_rqa_stat` dropdown."""
        
        for splice, rqa in zip([self.splice1, self.splice2, self.splice3], [self.sig1_rqa, self.sig2_rqa, self.sig3_rqa]):
            ts = splice.get_ydata()
            rp_instance, rp_matrix = get_rp_matrix(ts, self.spliced_rr_slider.value)
            rqa.value =  get_rqa_stat(rp_instance, self.spliced_rqa_stat.value, theiler=self.spliced_tau_slider.value*self.spliced_dim_slider.value, l_min=2, v_min=2)

        spliced_ts = self.splice_ts([self.splice1.get_ydata(), self.splice2.get_ydata(), self.splice3.get_ydata()])

        rp_instance, rp_matrix = get_rp_matrix(spliced_ts, self.spliced_rr_slider.value)
        self.full_spliced_rqa.value = get_rqa_stat(rp_instance, self.spliced_rqa_stat.value, theiler=self.spliced_tau_slider.value*self.spliced_dim_slider.value, l_min=2, v_min=2)

    def outline_splices(self, *args):
        """ Outline sub-RPs for each splice in the full RP."""

        if self.show_splices.value == True:
            for w in [self.window1, self.window2, self.window3]:
                w.set_visible(True)
        else:
            for w in [self.window1, self.window2, self.window3]:
                w.set_visible(False)
        self.redraw_rp()

    def splice_ts(self, spliced):

        """ Combine time series in `spliced` list/array into one light curve. """
        spliced_ts = np.append(spliced[0], spliced[1])
        spliced_ts = np.append(spliced_ts, spliced[2])
        
        return spliced_ts
    
    def format_slider_label(self, slider_val):
        num_decimals = str(slider_val)[::-1].find('.')
        is_int = slider_val%1 == 0
        fmt = "%d" if is_int else "%.1f" if num_decimals==1 else "%.3f" 
        return fmt % slider_val
    
    def create_default_fig(self, w, h, rows=1, cols=1, xlabel=None, ylabel=None, title=None, position=None, margins={'x':0.002, 'y':0.002}, ticks=True, labelticks=True, cbar=False, sc=None, layout=None, **kwargs):  
  
        try:
            plt.ioff()
        except: pass
        
        if layout:
            fig, ax = plt.subplots(figsize=(w,h), nrows=rows, ncols=cols, layout=layout)
        else:
            fig, ax = plt.subplots(figsize=(w,h), nrows=rows, ncols=cols)
        
        # turn autoscale on for ax
        ax.set_autoscale_on(True)
                               
        # see if matplotlib GUI backend in use
        try:
            plt.ioff()
            fig.canvas.toolbar_visible = False
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.resizable = False
        # if %matplotlib inline or %matplotlib notebook, skip
        except:
            pass

        if rows != 1 or cols != 1:
            subplots = True
        else:
            subplots = False

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14)
            
        if ylabel:
            if xlabel and ylabel == 'same':
                ax.set_ylabel(xlabel, fontsize=14)
            else:
                ax.set_ylabel(ylabel, fontsize=14)
        
        if title:
            if subplots and (isinstance(title, list) or isinstance(title, np.array)):
                for axis, t in zip(ax, title):
                    axis.set_title(title, fontsize=14)
            if not subplots:
                ax.set_title(title, fontsize=14)
            
        axes = ax if subplots else [ax]
        for axis in axes:
            # Set tick visibility based on the 'ticks' parameter
            axis.tick_params(which="both", axis="both", bottom=ticks, top=ticks, left=ticks, right=ticks,labelbottom=labelticks, labelleft=labelticks, labeltop=False, labelright=False, labelsize=14)

        if position and layout and layout != "constrained":
            fig.subplots_adjust(bottom=position['bottom'], top=position['top'], left=position['left'], right=position['right'])

        if margins:
            ax.margins(x=margins['x'], y=margins['y'])
            
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(sc, cax=cax)
            return fig, ax, cax, cb
        else:
            return fig, ax

def timedelay():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.transforms import Affine2D

    interval = 10
    ts = sine[:300]
    scatter_points_3d = np.array([[x, y, z] for x, y, z in zip(ts[::interval], ts[::interval][1:], ts[::interval][2:])])

    # Define the transformation matrix for isometric projection
    transform_matrix = np.array([[1, -0.5, 0],
                                [0, np.sqrt(3)/2, 0]])

    # Transformation function for isometric projection
    def isometric_projection(x, y, z):
        points_3d = np.array([x, y, z])
        points_2d = transform_matrix @ points_3d
        return points_2d

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(4,4))

    # Transform and plot each 3D point as a 2D scatter point
    for point in scatter_points_3d:
        x2d, y2d = isometric_projection(*point)
        ax.scatter(x2d, y2d, color='k', marker='o', zorder=5)

        # Draw rectangles representing the x, y, and z planes
    # for axis, color in zip(['x', 'y', 'z'], colors2_hex):
    #     rectangle = plt.Rectangle((-1, -1), 2, 2, edgecolor=color, alpha=0.2)
    #     ax.add_patch(rectangle)

    # for patch in ax.patches[-3:]:  # Assuming the last three patches are the rectangles
    #     patch.set_transform(Affine2D().rotate_deg(-60) + ax.transData)
        
    # Set aspect of the plot to be equal
    ax.set_aspect('equal')

    # Set limits and grid
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.axis('off')

    # Drawing arrows and annotating
    for label, vector in annotations.items():
        ax.annotate('', xy=vector, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="simple", fc=colors2_hex[0], ec="k", mutation_scale=20))
        ax.text(vector[0], vector[1], label, fontsize=14, ha='center', va='center')

    ax.plot(0,0,'ko', linestyle="")
    plt.savefig('rqa_tutorial/data/delay_anim.svg', bbox_inches='tight')
    plt.show()



from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import HTML

class GenerateRPs:
    
    def __init__(self):
        
        # parameters and variables
        self.rqa_vals = pd.read_csv('rqa/data/characteristic_rqa_stats_exclude_theiler.csv', index_col=0).to_dict(orient='index')
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
    

import matplotlib.pyplot as plt
plt.style.use('rqa_tutorial/data/standard.mplstyle') # need to use absolute path
plt.ioff()

import numpy as np

# %load_ext autoreload
# %autoreload 2

from traitlets import dlink
from matplotlib.patches import Rectangle
from rqa._utils import normalize, connect_splices, plot_rp_matrix, get_rp_matrix, update_extent, get_rqa_stat, validate_rqa #create_default_fig, format_slider_label
from ipywidgets import Layout, HBox, Box, VBox, IntSlider, FloatSlider, Dropdown, ToggleButton, ToggleButtons, Label, HTML
from rqa._styles import row, col, style
from pyunicorn.timeseries import RecurrencePlot
import seaborn as sns, pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
