class TimeDelayAnimation:
    def __init__(self):
        # Initialize parameters and widgets
        self.t = 0
        self.delta_t = 2
        self.m = 3
        self.scaling_factor = 10
        self.scatter_x = []
        self.scatter_y = []
        self.colors = plt.cm.cool(np.linspace(0, 1, dimension_slider.max))
        self.hex_colors_stripped = ['#dd1c77']
        self.extra_colors = [list(to_rgb(h.upper()))+[1] for h in self.hex_colors_stripped]
        
        self.delays = np.arange(0, 50*dimension_slider.value, 50)
        self.yticks = [7.5 + (2.5 * i) for i in range(3)] #dimension_slider.value)]

        self.frequency = 5  # Adjust frequency for 5 periods in the plot
        self.T = 10  # You can adjust this if you want to scale the time axis
        self.x_values = np.linspace(0, 2 * np.pi, 201)  # 200 points for the x_values
        self.sine_wave = np.sin(2 * np.pi * self.frequency * self.x_values / self.T)  # Adjusted sine_wave
        self.t_values = np.arange(0, 201)
        
        self.create_widgets()
        
        # Initialize static plot comparing x1, x2, x3
        plt.ioff()
        self.fig = plt.figure(figsize=(8, 5), layout='compressed')
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False
        self.ax = self.fig.add_subplot(1, 2, 1)
        self.ax.margins(x=0.02, y=0.2)
        self.ax.tick_params(which="both", axis="y", left=False, labelleft=True)
        self.ax.tick_params(which="both", axis="x", top=True)
        # set custom color cycle
        self.ax.set_prop_cycle('color', self.extra_colors + list(self.colors))
                # set initial tick labels
        self.ax.set_xlabel('Time (arb. units)')
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels([r'Original ($x_1$)'] + [r'$x_%d = x_1 + %d\tau$' % (i+2, i+1) if i != 0 else r'$x_2 = x_1 + \tau$' for i in range(2)]) #dimension_slider.value - 1)])
        

        # Initialize plot w/ animation
        plt.ioff()
        self.fig2 = plt.figure(figsize=(8, 10))
        self.fig2.canvas.toolbar_visible = False
        self.fig2.canvas.header_visible = False
        self.fig2.canvas.footer_visible = False
        self.fig2.canvas.resizable = False
        self.ax2 = self.fig.add_subplot(1, 2, 1)
        self.ax2.margins(x=0.02, y=0.2)
        self.ax2.tick_params(which="both", axis="both", left=False, labelleft=True, bottom=False, labelbottom=True)
        self.ax2.spines[['right', 'top']].set_visible(False)
        self.ax2.set_title(r'(x$_1$, x$_2$, x$_3$)')
            
        # 3D subplot
        self.ax3 = self.fig2.add_subplot(1, 2, 2, projection='3d')  
        self.ax3.set_title(r'State Space $S(\vec{x})$')
        self.ax3.margins(x=0.03, y=0.03)

        # set custom color cycle
        self.ax3.set_prop_cycle('color', self.extra_colors + list(self.colors))

        # Now that self.fig is defined, plot initial state and connect events
        self.init_plot()
        self.connect_events()
        
        # Initialize animation
        self.total_frames = self.calculate_total_frames()  # Calculate total frames
        self.ani = FuncAnimation(self.fig, self.update, frames=self.total_frames,
                                interval=1000, init_func=self.init_plot, blit=False, repeat=True)
        plt.show()
        
        
    def init_plot(self):
        """  Sets up the initial state of the plot """
        
        # Initialize a list to store the 3D scatter points
        self.scatter_points_3d = []

        # Clear the axes for the reset
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot sine wave
        self.ax2.plot(self.sine_wave) #self.x_values, self.sine_wave)
        
        # Configure x-axis major ticks
        self.ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
        
        # Plot y=0 line
        for y in [-1, 0, 1]: self.ax2.axhline(y=y, color='grey', alpha=0.3, linestyle=':')
    
        # Initialize scatter points for sine wave
        self.scatter = self.ax2.scatter([], [], c=[], zorder=3)
        self.vlines = self.ax2.vlines([], 0, [], zorder=2)
        
        # Initialize 3D scatter
        self.scatter3d = self.ax3.scatter([], [], [], c='black')
        
        # Initialize annotations
        self.annotations = [self.ax2.annotate('', (0,0), textcoords="offset points", xytext=(5,5), ha='right') for _ in range(3)]
        
        # Set plot limits and labels
        self.ax2.set_ylim(-1.1, 1.1)
        
        self.ax3.set_xlim(-1.1, 1.1)
        self.ax3.set_ylim(-1.1, 1.1)
        self.ax3.set_zlim(-1.1, 1.1)
        
        self.ax2.set_xlabel(r'$t_{rest}$ (arb. units)')
        self.ax2.set_ylabel('Amplitude')
        title = r'$\mathbf{\vec{x}}=$(' + ", ".join([r"$x_%d$" % (i+1) for i in range(3) ]) + r")"
        time = r"$\mathbf{t_{rest}=0}$"
        title1 = self.ax2.set_title(title, fontsize=14, loc='center')
        title2 = self.ax2.set_title(time, fontsize=14, loc='right', color='r')
        
         # Return a list of artists to be drawn
        return self.scatter, self.vlines, self.scatter3d, *self.annotations
    
       
    def update(self, frame):
        """ Update scatter points, sine wave, annotations, and 3D scatter """
        self.t = (int(frame) * self.scaling_factor) % len(self.sine_wave)

        # Ensure delta_t is an integer and apply the scaling factor
        delta_t_scaled = int(self.delta_t) * self.scaling_factor

        # Calculate the indices for y-values without the modulus operator
        # The modulus is not needed here if we are sure that frame will not produce an out-of-bounds index
        idx1 = self.t
        idx2 = self.t + delta_t_scaled
        idx3 = self.t + 2 * delta_t_scaled
            
        # Get the y-values for the 3D scatter plot
        y1 = self.sine_wave[idx1]
        y2 = self.sine_wave[idx2]
        y3 = self.sine_wave[idx3]
        self.scatter_x.append(idx3)
        self.scatter_y.append(y3)

        colors = plt.cm.jet(np.linspace(0, 1, self.m))
        
        # Update 2D scatter plot
        x_scatter = self.t_values[[idx1, idx2, idx3][:self.m]]  # Get x-values for scatter points
        y_scatter = [y1, y2, y3][:self.m]  # Get y-values for scatter points
        self.scatter.set_offsets(np.column_stack([x_scatter, y_scatter]))
        self.scatter.set_color(colors)
        
        # Update vertical lines
        self.vlines.set_segments([[[x, 0], [x, y]] for x, y in zip(x_scatter, y_scatter)])
        self.vlines.set_color(colors)
        
        # Update 3D scatter plot by appendinge new point to the list of scatter points
        
        if self.m == 3:
            self.scatter_points_3d.append([y1, y2, y3])
        elif self.m == 2:
            self.scatter_points_3d.append([y1, y2, 0])
        elif self.m == 1:
            self.scatter_points_3d.append([y1, 0, 0]) 
            
        # Clear the previous scatter plot and redraw with all points
        self.ax3.clear()
        self.ax3.scatter(*zip(*self.scatter_points_3d))

        # Update annotations
        for i, (x, y) in enumerate(zip(x_scatter, y_scatter)):
            if i < self.m:
                self.annotations[i].set_text(f'y{i+1}')
                self.annotations[i].xy = (x, y)
                self.annotations[i].set_visible(True)
                # Set the position, text, visibility, etc. for each annotation
            else:
                self.annotations[i].set_visible(False)
        
         # Return a list of artists to be redrawn
        return self.scatter, self.vlines, self.scatter3d, *self.annotations
    
    
    def reset_anim(self):
        # Reset the animation when interacting with the widgets
        self.t = 0
        # self.pause_anim()
        # double check that we Clear scatter points and reset annotations

        self.scatter.set_offsets([])
        self.scatter3d.set_data_3d([], [], [])

        self.total_frames = self.calculate_total_frames()  # Recalculate total frames
        self.ani = FuncAnimation(self.fig, self.update, frames=self.total_frames,
                                interval=1000, init_func=self.init_plot, blit=False, repeat=True)
       
    def restart_anim(self):
        # Restart the animation after interaction
        self.ani.event_source.start()
    
    def resume_anim(self):
        # Restart the animation after interaction
        self.ani.event_source.start()
    
    def pause_anim(self):
        # Pause the animation
        self.ani.event_source.stop()
    
    def update_delta_t(self, change):
        # Update delta_t based on the slider
        self.delta_t = change.new
        self.reset_anim()
    
    def update_m(self, change):
        # Update m based on the slider
        self.m = change.new
        self.reset_anim()
        
    def calculate_total_frames(self):
        # Calculate the last valid start index for t
        # This index is the one that allows the last y value to be the last point in sine_wave
        last_valid_start = len(self.sine_wave) - (self.m - 1) * self.delta_t * self.scaling_factor
        
        # Calculate the total number of frames
        # Divide by the step size and add one to account for the first frame
        total_frames = last_valid_start // (self.scaling_factor) + 1
        self.total_frames = total_frames
        return total_frames
    
    def dynamic_annotation(self, i, x, y):
        """ Dynamically place annotations based on the y-value and slope """
        
        # Calculate slope of the sine wave at x
        slope = np.cos(2 * np.pi * self.frequency * x / self.T)
        angle = np.arctan(slope) * 180 / np.pi
        
        # Adjust position based on the slope and y-value
        offset_x = 0.15 * np.cos(angle)
        offset_y = 0.15 * np.sin(angle)
        if y > 0 and slope > 0:
            # Annotations on the right for positive slope and positive y
            ha, va = 'left', 'bottom'
        
        elif y > 0:
            # Annotations on the left for negative slope and positive y
            ha, va = 'right', 'top'
        elif slope > 0:
            # Annotations on the left for positive slope and negative y
            ha, va = 'right', 'bottom'
        elif y == 1:
            offset_x, offset_y = 0, 0.15 # override offset
            ha, va = 'left', 'top'
        elif y == -1:
            offset_x, offset_y = 0, -0.15 # override offset
            ha, va = 'left', 'bottom'
        else:
            # Annotations on the right for negative slope and negative y
            ha, va = 'left', 'top'
        
        self.annotations[i].set_x(x + offset_x)
        self.annotations[i].set_y(y + offset_y)
        self.annotations[i].set_ha(ha)
        self.annotations[i].set_va(va)
    
    def create_widgets(self):
        # Create and setup ipywidgets for delta_t and m
        self.delta_t_slider = widgets.IntSlider(min=2, max=20, value=self.delta_t, description='Delta T:')
        self.m_slider = widgets.IntSlider(min=1, max=3, value=self.m, description='M:')
        display(self.delta_t_slider, self.m_slider)
    
    def connect_events(self):
        # Attach event handlers to the widgets
        self.delta_t_slider.observe(self.update_delta_t, names='value')
        self.m_slider.observe(self.update_m, names='value')
        
         # ipywidget events
#         self.d1 = ipyevents.Event(source=self.delta_t_slider, watched_events=['mouseup', 'mousedown'], wait=500, throttle_or_debounce='debounce')
#         self.d2 = ipyevents.Event(source=self.m_slider, watched_events=['mouseup', 'mousedown'], wait=500, throttle_or_debounce='debounce')
        # self.d3 = ipyevents.Event(source=self.show_trest, watched_events=['mouseup', 'mousedown'], wait=500, throttle_or_debounce='debounce')

        # connect events to DOM event handler
        for event in [self.d1, self.d2]: event.on_dom_event(self.toggle_pause)
        
        
        # bind pause_anim to mouse clicks
        self.fig.canvas.mpl_connect('button_press_event', pause_anim)

    def toggle_pause(self, event):
        """ Event handler for widget interaction / DOM callbacks."""  
        
         # once user clicks down on a widget for interaction, pause animation
        if event['type'] == 'mousedown': 
            self.pause_anim()
        # resume animation on mouse release
        elif event['type'] == 'mouseup': 
            self.resume_anim()