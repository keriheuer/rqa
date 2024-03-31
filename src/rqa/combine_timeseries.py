from .config import *

class CombineTimeseries():

    def __init__(self):

        self.cmap = cmap 
        self.systems = systems
        self.stats = stats 
        self.signals_ts = [white_noise, sine, super_sine, logi, brownian]
        self.colors, self.timeseries = colors, ts_data
        self.rqa_vals = characteristic_rqa_stats
                
        self.ts1_type = Dropdown(options=self.systems, value='Brownian Motion', layout=Layout(width="200px", font_size="16px"))
        self.ts2_type = Dropdown(options=self.systems, value='Sine', layout=Layout(width="200px", font_size="16px"))
        self.ts3_type = Dropdown(options=self.systems, value='White Noise', layout=Layout(width="200px", font_size="16px"))
        
        self.ts1_amp = HBox([FloatSlider(value=1, min=0.05, max=2, step=0.01, readout=False, continuous_update=False, description='𝐴₁', layout=Layout(width="150px", height="25px", overflow="visible", font_size="16px")),
                             Label("1", layout=Layout(width="fit-content", font_size="16px"))])
        self.ts2_amp = HBox([FloatSlider(value=1, min=0.05, max=2, step=0.01, readout=False, continuous_update=False, description='𝐴₂', layout=Layout(width="150px", height="25px", overflow="visible", font_size="16px")),
                             Label("1", layout=Layout(width="fit-content", font_size="16px"))])
        self.ts3_amp = HBox([FloatSlider(value=1, min=0.05, max=2, step=0.01, readout=False, continuous_update=False, description='𝐴\u2083', layout=Layout(width="150px", height="25px", overflow="visible", font_size="16px")),
                             Label("1", layout=Layout(width="fit-content", font_size="16px"))])
        
        self.ts1_duration = HBox([IntSlider(value=300, min=200, max=1000, step=10, readout=False, continuous_update=False, description='𝑇₁', layout=Layout(width="150px", height="25px", overflow="visible", font_size="16px")), Label("200", layout=Layout(width="fit-content", font_size="16px"))])
        self.ts2_duration = HBox([IntSlider(value=300, min=200, max=1000, step=10, readout=False, continuous_update=False, description='𝑇₂', layout=Layout(width="150px", height="25px", overflow="visible", font_size="16px")), Label("200", layout=Layout(width="fit-content", font_size="16px"))])
        self.ts3_duration = HBox([IntSlider(value=300, min=200, max=1000, step=10, readout=False, continuous_update=False, description='𝑇\u2083', layout=Layout(width="150px", height="25px", overflow="visible", font_size="16px")), Label("200", layout=Layout(width="fit-content", font_size="16px"))])

        self.ts1 = VBox([self.ts1_type, self.ts1_amp, self.ts1_duration], layout=Layout(display="flex", flex_direction="column", align_items="center", width="33.33%", justify_content="space-between"))
        self.ts2 = VBox([self.ts2_type, self.ts2_amp, self.ts2_duration], layout=Layout(display="flex", flex_direction="column", align_items="center", width="33.33%", justify_content="space-between"))
        self.ts3 = VBox([self.ts3_type, self.ts3_amp, self.ts3_duration], layout=Layout(display="flex", flex_direction="column", align_items="center", width="33.33%", justify_content="space-between"))

        self.ts_list = HBox([self.ts1, self.ts2, self.ts3], layout=Layout(display="flex", flex_direction="row", height="110px", top="0%", align_self="flex-end", width="90%"))
        self.ts_list.layout.margin = "0 0 15px 0"
        
        self.spliced_rr_slider = FloatSlider(min=0, max=0.25, value=0.1, step=0.01, continuous_update=False, readout=False, layout=Layout(width="150px", overflow="visible", height="25px", ))
        self.spliced_rr_label = Label("0.10", layout=Layout(width="fit-content"))
        self.spliced_rr = HBox([Label('𝑅𝑅'), self.spliced_rr_slider, self.spliced_rr_label])
        
        self.show_splices = ToggleButton(value=False, description="Outline RP windows", layout=Layout(width="auto"))

        self.spliced_tau_slider = IntSlider(min=1, max=20, value=1, step=1, continuous_update=False, description="𝜏  ", readout=False, layout=Layout(width="150px", overflow="visible", height="25px", font_size="16px"))
        self.spliced_tau_label = Label("1", layout=Layout(width="fit-content", font_size="16px"))
        self.spliced_tau = HBox([self.spliced_tau_slider, self.spliced_tau_label])
  
        self.spliced_dim_slider =  IntSlider(min=1, max=8, value=1, step=1, continuous_update=False, description="𝑚  ", readout=False, layout=Layout(width="150px", overflow="visible", height="25px", font_size="16px"))
        self.spliced_dim_label = Label("1", layout=Layout(width="fit-content", font_size="16px"))
        self.spliced_dim = HBox([self.spliced_dim_slider, self.spliced_dim_label])
        
        self.spliced_rp_ui_left = VBox([self.spliced_rr, self.show_splices])
        self.spliced_rp_ui_left.layout.margin = "0 0 0 25px"
        self.spliced_rp_ui_right = VBox([self.spliced_tau, self.spliced_dim])
        self.spliced_rp_ui_right.layout.margin = "0 10px 0 0"
        
        self.spliced_rp_ui = HBox([self.spliced_rp_ui_left, self.spliced_rp_ui_right], layout=Layout(display="flex", flex_direction="row", justify_content="space-between", height="60px", right="0", align_self="stretch", overflow="visible"))
        self.spliced_rp_ui.layout.margin = "0 0 15px 0"
        
        self.spliced_rqa_stat = Dropdown(options=['DET', 'LAM', 'RR', 'L MAX', 'L MEAN', 'L ENTR', 'DIV', 'V MAX', 'TT',  'V ENTR'], value='DET',
                                            layout=Layout(width="150px", align_self="flex-end", bottom="0", font_size="16px"))
        
        self.sig1 = HTML(value='<p style="font-size: 18px"><font color="%s">' % self.colors[self.ts1_type.value] + '%s' % self.ts1_type.value + '</font></p>', layout=Layout(font_size="18px"))
        self.sig2 = HTML(value='<p style="font-size: 18px"><font color="%s">' % self.colors[self.ts2_type.value] + '%s' % self.ts2_type.value + '</font></p>', layout=Layout(font_size="18px"))
        self.sig3 = HTML(value='<p style="font-size: 18px"><font color="%s">' % self.colors[self.ts3_type.value] + '%s' % self.ts3_type.value + '</font></p>', layout=Layout(font_size="18px"))
        
        self.sig1_rqa = HTML("", style=style, continuous_update=False, layout=Layout(font_size="18px"))
        self.sig2_rqa = HTML("", style=style, continuous_update=False, layout=Layout(font_size="18px"))
        self.sig3_rqa = HTML("", style=style, continuous_update=False, layout=Layout(font_size="18px"))
        
        self.full_spliced = HTML('<p style="font-size: 18px">Combined</p>', style=style, layout=Layout(font_size="18px"))
        self.full_spliced_rqa = HTML("", style=style, continuous_update=False, layout=Layout(font_size="18px"))
        
        for w in [self.sig1_rqa, self.sig2_rqa, self.sig3_rqa, self.full_spliced_rqa]:
            w.layout.margin = "10px 0 0 0"

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
        self.sig1_rqa.value = '<p style="font-size: 18px; margin-top: 10px">%s</p>' % rqa_val
        
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
        self.sig2_rqa.value = '<p style="font-size: 18px; margin-top: 10px">%s</p>' % rqa_val
        

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
        self.sig3_rqa.value = '<p style="font-size: 18px; margin-top: 10px">%s</p>' % rqa_val

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