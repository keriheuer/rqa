from matplotlib.backend_tools import Cursors
from .config import *

        
class CharacteristicRQA:
    
    def __init__(self):
        plt.ioff()
        
        # create characteristic RP plots
        self.zoomed = False
        self.systems = np.array(['White Noise', 'Sine', 'Sinusoid', 'Logistic Map', 'Brownian Motion'])
        self.colors = ['#db57b2', '#a157db', '#5770db', '#57d3db', '#57db80']
        rps = self.characteristic_rps()
    
        # create characteristic RQA plot
        self.rqa_vals = pd.read_csv(get_resource_path('data/characteristic_rqa_stats_exclude_theiler.csv'), index_col=0).to_dict(orient='index')

        self.stats = {'DET': 'Determinism', 'LAM': 'Laminarity', 'L MAX': 'Longest Diagonal Line Length', 
                      'V MAX': 'Longest Vertical Line Length', 'L MEAN': 'Average Diagonal Line Length', 
                      'TT': 'Trapping Time', 'DIV': 'Divergence (1/L MAX)', 'L ENTR': 'Diagonal Line Entropy', 'V ENTR': 'Vertical Line Entropy',
                      'C ENTR': 'Complexity Entropy', 'T REC': 'Mean Recurrence Time'}
        
        dropdown_layout = Layout(max_width="200px", overflow="visible")
        self.rqa_stat = Dropdown(options=list(self.stats.keys()), layout=dropdown_layout)
        
        rqa = self.create_rqa_bar_plot()

        # render Dropdown and bar plot vertically
        display(HBox([rps, VBox([self.rqa_stat, rqa], 
                                       layout=Layout(display="flex", flex_direction="column", align_items="center"))]))
        
    def characteristic_rps(self):
         
        # load in pre-generated time series data
        self.white_noise = np.load(get_resource_path("data/example_timeseries/white_noise_ts.npy"))
        self.sine = np.load(get_resource_path("data/example_timeseries/sine_ts.npy"))
        self.super_sine = np.load(get_resource_path("data/example_timeseries/super_sine_ts.npy"))
        self.logi = np.load(get_resource_path("data/example_timeseries/logistic_map_ts.npy"))
        self.brownian = np.load(get_resource_path("data/example_timeseries/brownian_ts.npy"))
        
        # load in corresponding RPs
        self.white_noise_rp = np.load(get_resource_path("data/example_rps/white_noise_rp.npy"))
        self.sine_rp = np.load(get_resource_path("data/example_rps/sine_rp.npy"))
        self.super_sine_rp = np.load(get_resource_path("data/example_rps/super_sine_rp.npy"))
        self.logi_rp = np.load(get_resource_path("data/example_rps/logistic_map_rp.npy"))
        self.brownian_rp = np.load(get_resource_path("data/example_rps/brownian_rp.npy"))
      
        self.signals_ts = [self.white_noise, self.sine, self.super_sine, self.logi, self.brownian]
        self.signals_rp = [self.white_noise_rp, self.sine_rp, self.super_sine_rp, self.logi_rp, self.brownian_rp]

        # create figure
        self.fig, self.axs = plt.subplots(nrows=2, ncols=5, figsize=(8.5, 4), layout='compressed')
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False

        # light curve plots
        for i, axis in enumerate(plt.gcf().get_axes()[:5]):
            if i==0: ts, = axis.plot(self.signals_ts[i][:100], linewidth=1, color=self.colors[i])
            elif i==3: ts, = axis.plot(self.signals_ts[i][:50], linewidth=1, color=self.colors[i])
            else: ts, = axis.plot(self.signals_ts[i][:300], linewidth=1, color=self.colors[i])
            axis.set_aspect(1./axis.get_data_ratio())
            axis.set_title(self.systems[i], fontsize=14)
            axis.cursor_to_use = Cursors.POINTER
            ts.set_gid(i)
            
        # RP plots
        for i, axis in enumerate(plt.gcf().get_axes()[5:]): 
            rp = axis.matshow(self.signals_rp[i], origin='lower', cmap='Greys', interpolation='none', picker=True)
            rp.set_gid(i)
            axis.set_aspect('equal')
            axis.cursor_to_use = Cursors.HAND
            
        # turn off ticks
        for axis in plt.gcf().get_axes():
            axis.tick_params(which="both", axis="both", left=False, right=False, top=False, bottom=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
        
        # connect figure to event handlers
        self.cid = self.fig.canvas.mpl_connect('figure_leave_event', self.on_leave_figure)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid3 = self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

        return self.fig.canvas

    def create_rqa_bar_plot(self):
        
        self.ticks = np.arange(len(self.systems))
        self.width = 0.25  # the width of the bars
        self.offset = self.width/2

        # create bar plot for RQA stats
        self.rqa_bar_plot, self.rqa_bar_plot_ax = plt.subplots(figsize=(4.5,5.5))
        self.rqa_bar_plot.canvas.toolbar_visible = False
        self.rqa_bar_plot.canvas.header_visible = False
        self.rqa_bar_plot.canvas.footer_visible = False
        self.rqa_bar_plot.canvas.resizable = False

        vals = list(self.rqa_vals['DET'].values())
        self.bars = self.rqa_bar_plot_ax.bar(self.ticks, vals, width=self.width, label=self.systems, color=self.colors)
        self.rqa_bar_plot_ax.bar_label(self.bars, padding=3, fontsize=12, fmt='{:.3f}')

        self.rqa_bar_plot_ax.set_ylabel('Fraction of Recurrences', fontsize=14)
        self.rqa_bar_plot_ax.set_title('Determinism', fontsize=14)
        self.rqa_bar_plot_ax.tick_params(which="both", axis="both", top=False, right=True, bottom=False, left=True, direction="in", labelsize=14)
        self.rqa_bar_plot_ax.set_xticks(self.ticks+self.offset, self.systems, rotation=50, horizontalalignment='right')
        self.rqa_bar_plot_ax.margins(x=0.2, y=0.15)

        # hide any ticks over the max value that show bc of added margin
        y_pos = self.rqa_bar_plot_ax.yaxis.get_minorticklocs()
        y_ticks = np.array(self.rqa_bar_plot_ax.yaxis.get_minor_ticks())
        idx = np.where(y_pos > np.max(vals))[0]
        for y in y_ticks[idx]:
            y.set_visible(False)
            
        # adjust canvas position
        plt.subplots_adjust(left=0.2, bottom=0.25, right=1)
       
        # register callbacks
        self.rqa_stat.observe(self.update_bars, type="change") # names="value")
        
        return self.rqa_bar_plot.canvas
    
    
    ## CALLBACKS
    
    def on_leave_figure(self, event):
        
        # if inset axes are visible and mouse leaves figure, close zoomed RP
        
        if self.zoomed:
            
            self.big_ax.remove()
            self.zoomed = False
            
            event.canvas.draw_idle()
            event.canvas.flush_events()
        
    def on_press(self, event):
        
        # if no zoomed RP is shown and click was inside an axes
        if not self.zoomed and event.inaxes:
            
            # then check if click was on RP plot or light curve
            contains_rp = event.inaxes.findobj(lambda x: isinstance(x,mpl.image.AxesImage), include_self=False) != []
            
            if contains_rp:
                
                # get id of RP clicked on
                id = event.inaxes.images[0].get_gid()
                
                # add inset axes and plot that RP
                self.big_ax = event.canvas.figure.add_axes([0.02,0.02,0.98,0.97], zorder=2)
                self.big_ax.matshow(self.signals_rp[id], cmap="Greys", origin='lower', interpolation='none')
                self.big_ax.tick_params(which="both", axis="both", left=False, right=False, top=False, bottom=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)

                # toggle zoomed attribute
                self.zoomed = True
                    
        # if click occurs anywhere in figure and inset axes are shown, remove 
        elif self.zoomed and event.canvas.figure.contains(event):
            
            self.big_ax.remove()
            self.zoomed = False
            
        # flush fig
        event.canvas.draw_idle() 
        event.canvas.flush_events()
        
    def on_hover(self, event):
        """ Change the cursor type for hovering over time series vs. RPs."""
   
        # only update cursor if inset axes not shown
        if not self.zoomed:
            self.fig.canvas.set_cursor(event.inaxes.cursor_to_use if event.inaxes else Cursors.POINTER)
            
        else:
            # set cursor=hand for whole figure
            self.fig.canvas.set_cursor(Cursors.HAND)
        
    def update_bars(self, *args):

        if self.rqa_stat.value in ['L MAX', 'V MAX', 'T REC']:
            labeltemplate = '{:.0f}'
            ylabel = 'Time (arb. units)'
            
        elif self.rqa_stat.value in ['TT', 'L MEAN']: # average lengths so could have decimals
            labeltemplate = '{:.2f}'
            ylabel = 'Time (arb. units)'
            
        elif self.rqa_stat.value == 'DIV':
            labeltemplate = '{:.2f}'
            ylabel = '1 / Time (arb. units)'
        elif 'ENTR' in self.rqa_stat.value:
            labeltemplate = lambda x: '%.2f' % x if x >= 0.01 else format_sci(x)
            ylabel = "Probability (Unitless)"
        else:
            labeltemplate = lambda x: '%.3f' % x if x >= 0.01 else format_sci(x)
            ylabel = 'Fraction of Recurrences'

        # # clear axes
        self.rqa_bar_plot_ax.cla()

        # redraw bars
        new_stats = list(self.rqa_vals[self.rqa_stat.value].values())
        self.rqa_bar_plot_ax.set(ylabel=ylabel, title=self.stats[self.rqa_stat.value], ylim=(0, np.max(new_stats) * 1.1))
        bars = self.rqa_bar_plot_ax.bar(self.ticks, new_stats, width=self.width, label=self.systems, color=self.colors)
        self.rqa_bar_plot_ax.bar_label(bars, fmt=labeltemplate, padding=3, fontsize=12)
        self.refresh_bars()
            
        # adjust plot
        self.rqa_bar_plot_ax.tick_params(which="both", axis="both", top=False, right=True, bottom=False, left=True, labelsize=14)
        self.rqa_bar_plot_ax.set_xticks(self.ticks+self.offset, self.systems, rotation=50, horizontalalignment='right') 
        self.rqa_bar_plot_ax.margins(x=0.2, y=0.15)
        self.refresh_bars()
        
    def refresh_bars(self):
        # flush fig
        self.rqa_bar_plot.canvas.draw_idle()
        self.rqa_bar_plot.canvas.flush_events()
        