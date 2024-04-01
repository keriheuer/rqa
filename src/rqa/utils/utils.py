col = Layout(display="flex", flex_direction="column", align_items="center", width="33.33%")
row = Layout(display="flex", flex_direction="row", width="100%", align_self="stretch", left="0", height="auto", text_align='left')
col_align_left = Layout(display="flex", flex_direction="row", width="100%", align_self="flex-start", left="0", height="60px", text_align='center', align_items="center")

widget_layout = Layout(width='100%', height='25px', margin="5px")

button_layout = Layout(width='auto', max_width="250px", height='25px', margin="10px")

ui_layout = Layout(display="flex", height="350px", flex_flow="column", overflow="visible",
    align_items="center", justify_content="center", width="45%")

horizontal_layout = Layout(display="flex", flex_flow="row", height="350px", overflow= "visible",
    align_items="center", justify_content="center")

vertical_layout = Layout(display="flex", flex_flow="column", height="350px", overflow= "visible",
    align_items="center", justify_content="center")

label_layout = Layout(display="flex", flex_flow="row", align_items="center", justify_content="center", overflow="visible",
    width="100%", text_align="center", height='40px', margin="10px")

slider_layout = Layout(min_width="350px", overflow="visible")

dropdown_layout = Layout(max_width="200px", overflow="visible")

style = {'description_width': 'initial', "button_width": "auto", 'overflow':'visible',
            'white-space': 'nowrap', 'button_color': 'lemonchiffon', 'handle_color': 'cornflowerblue'}

# explicitly set width and height
layout = Layout(width='auto', height='25px', display="flex", margin="10px")

# function to get the full path of files in the package
def get_resource_path(relative_path):
    return pkg_resources.resource_filename('rqa', relative_path)
    # return importlib.resources.files("rqa")+relative_path

# class Annotation3D(Annotation):

#     def __init__(self, text, xyz, *args, **kwargs):
#         super().__init__(text, xy=(0, 0), *args, **kwargs)
#         self._xyz = xyz

#     def draw(self, renderer):
#         x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
#         self.xy = (x2, y2)
#         super().draw(renderer)
# # For seamless integration we add the annotate3D method to the Axes3D class.

# def _annotate3D(ax, text, xyz, *args, **kwargs):
#     '''Add anotation `text` to an `Axes3d` instance.'''

#     annotation = Annotation3D(text, xyz, *args, **kwargs)
#     ax.add_artist(annotation)
#     return annotation

# class Arrow3D(FancyArrowPatch):

#     def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._xyz = (x, y, z)
#         self._dxdydz = (dx, dy, dz)

#     def draw(self, renderer):
#         x1, y1, z1 = self._xyz
#         dx, dy, dz = self._dxdydz
#         x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

#         xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         super().draw(renderer)

#     def do_3d_projection(self, renderer=None):
#         x1, y1, z1 = self._xyz
#         dx, dy, dz = self._dxdydz
#         x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

#         xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

#         return np.min(zs)

# def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
#     '''Add an 3d arrow to an `Axes3D` instance.'''

#     arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
#     ax.add_artist(arrow)
#     return arrow


def find_longest_diag(array2D, patternlength):
    res=[]
    for k in range(-array2D.shape[0]+1, array2D.shape[1]):
        diag=np.diag(array2D, k=k)
        if(len(diag)>=patternlength):
            for i in range(len(diag)-patternlength+1):
                if(all(diag[i:i+patternlength]==1)):
                    res.append((i+abs(k), i) if k<0 else (i, i+abs(k)))
    return res

def find_vmax(distances, i, vmax):
  # find start positions of all vertical lines with length = V MAXß
  col = distances[i, :]
  idx_pairs = np.where(np.diff(np.hstack(([False],col==1,[False]))))[0].reshape(-1,2)
  lengths = np.diff(idx_pairs,axis=1) # island lengths
  if vmax in lengths:
    return (i, idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]) # longest island start position


def find_lmax(array2D, lmax):
  # find start position of diagonal with length = L MAX
    res = []
    for k in range(-array2D.shape[0]+1, array2D.shape[1]):
        diag = np.diag(array2D, k=k)
        if(len(diag) >= lmax):
            for i in range(len(diag) - lmax + 1):
                if(all(diag[i:i + lmax] == 1)):
                    res.append((i + abs(k), i) if k<0 else (i, i + abs(k)))
    return res

def transform_to_area(x, y1, y2):

    """ Helper function for filling between lines in bqplot figure."""

    return np.append(x, x[::-1]), np.append(y1, y2[::-1])

def exclude_diagonals(rp, theiler):

  # copy recurrence matrix for plotting
  rp_matrix = rp.recurrence_matrix()

  # clear pyunicorn cache for RQA
  rp.clear_cache(irreversible=True)

  # set width of Theiler window (n=1 for main diagonal only)
  n = theiler

  # create mask to access corresponding entries of recurrence matrix
  mask = np.zeros_like(rp.R, dtype=bool)

  for i in range(len(rp_matrix)):
    mask[i:i+n, i:i+n] = True

  # set diagonal (and subdiagonals) of recurrence matrix to zero
  # by directly accessing the recurrence matrix rp.R

  rp.R[mask] = 0
  
def connect_splices(spliced):

  spliced[0][0] = 0
  spliced[0][-1] = 0
  spliced[1][0] = 0
  spliced[1][-1] = 0
  spliced[2][0] = 0
  spliced[2][-1] = 0

  return spliced

def normalize(x, name):

  x = x.astype(float)

  z = (x - np.mean(x))/np.std(x)

  # scale factor for white noise time series
  if name == 'White Noise':
    z *= 0.55

  return z
  
def create_fig(w, h, rows=1, cols=1, xlabel="", ylabel="", title="", position=None, margins=None, ticks=True, aspect='auto', sizes=None, layout=None, **kwargs):  
  
  if layout:
    fig, ax = plt.subplots(figsize=(w,h), nrows=rows, ncols=cols, layout=layout)
  else: 
     fig, ax = plt.subplots(figsize=(w,h), nrows=rows, ncols=cols)
     
  # see if matplotlib GUI backend in use
  try:
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.resizable = False
  except:
    pass
    
  if (rows != 1) or (cols != 1): # make subplots
      for axis, ax_title in zip(ax, title):
        axis.set(ylabel=ylabel, xlabel=xlabel, autoscale_on=True, aspect=aspect)
        axis.set_title(ax_title, fontsize=16)
  else:
    ax.set(ylabel=ylabel, xlabel=xlabel, autoscale_on=True, aspect=aspect)
    ax.set_title(title, fontsize=16)
  
  if ticks:
    ax.tick_params(which="both", axis="both", bottom=True, top=True, left=True, right=True, labeltop=False, labelbottom=True, labelleft=True, labelsize=14)
  else:
    ax.tick_params(which="both", axis="both", bottom=False, top=False, left=False, right=False, labeltop=False, labelbottom=True, labelleft=True, labelsize=14)

  if position and not layout:
    plt.subplots_adjust(bottom=position['bottom'], top=position['top'], left=position['left'], right=position['right'])

  if margins:
    ax.margins(x=margins['x'], y=margins['y'])
    
  return fig, ax

def add_colorbar(fig, ax, sc, cbar_title="", **kwargs):
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  cb = fig.colorbar(sc, cax=cax)
  cb.set_label(label=cbar_title,fontsize=16)
  return cb

def sync_rp(fig, ax, sc, data):

    ax.tick_params(which="both", axis="both", right=False, bottom=False, top=False, left=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    update_extent(sc, data)
    fig.draw_idle()

def splice_timeseries(self):
    # normalize the data to same scale
  spliced = [self.brownian, self.sine, self.white_noise]
  spliced = [normalize(s[:500], n) for s,n in zip(spliced, ['Brownian', 'Sine', 'White Noise'])]

  # connect end points of time series
  spliced = connect_splices(spliced)
  return spliced


def get_rp_matrix(ts, RR, tau=1, dim=1):
  rp = RecurrencePlot(ts, metric="euclidean", recurrence_rate=RR, silence_level=2, tau=tau, dim=dim)
  return rp, rp.recurrence_matrix()

def plot_rp_matrix(ax, rp):
  rp_matrix = ax.matshow(rp, cmap='Greys', origin='lower', interpolation='none')
  ax.tick_params(which="both", axis="both", bottom=False, top=False, left=False, right=False, labeltop=False, labelbottom=True, labelleft=True)
  update_extent(rp_matrix, rp)
  return rp_matrix


def format_slider_label(slider_val):
  num_decimals = str(slider_val)[::-1].find('.')
  is_int = slider_val%1 == 0
  fmt = "%d" if is_int else "%.1f" if num_decimals==1 else "%.3f" 
  return fmt % slider_val

def format_sci(x):
  """ Nicely format number into scientific notation string, i.e. 1e-4.
  x: float"""
  
  # split sci notation string by e-
  if x < 1:
    delimiter = "e-"
    
  # split sci notation string by e+
  elif x > 10:
     delimiter = "e+"
     
  number = "%.1e" % x  # 1.0e-04
  lead, power = number.split(delimiter)
  
  # strip any leading zeros from exponent
  power = power.lstrip('0')
  
  # strip trailing .0 if lead is integer
  lead = lead.rstrip('.0')
  
  return lead + delimiter + power

def format_rqa_stat(stat, val):
    if 'ENTR' in stat: 
      fmt = "%.2f"
    elif stat in ['DET', 'RR', 'DIV', 'LAM', 'TT']: 
      fmt = "%.2f"
    elif ('MEAN' in stat) or ('MAX' in stat):
      fmt = "%d"
      
    return fmt % str(val)
  
def get_rqa_stat(rp, stat, theiler=1, l_min=2, v_min=2):
  
  # exclude Theiler window if stat is diagonal-line based 
  if stat in ['DET', 'L MAX', 'L MEAN', 'DIV', 'L ENTR']:
    exclude_diagonals(rp,theiler)
  
  rqa_vals = {'DET': validate_rqa("%.3f", rp.determinism(l_min=l_min)), 
              'LAM': validate_rqa("%.3f", rp.laminarity(v_min=v_min)), 
              'L MAX': validate_rqa("%d", rp.max_diaglength()), 
              'L MEAN': validate_rqa("%.2f", rp.average_diaglength(l_min=l_min)), 
              'V MAX': validate_rqa("%d", rp.max_vertlength()), 
              'V MEAN': validate_rqa("%.2f", rp.average_vertlength(v_min=v_min)), 
              'L ENTR': validate_rqa("%.3f", rp.diag_entropy(l_min=l_min)), 
              'V ENTR': validate_rqa("%.3f", rp.vert_entropy(v_min=v_min)), 
              'DIV': validate_rqa("%.3f", (1/rp.max_diaglength())), 
              'TT': validate_rqa("%.3f", rp.trapping_time(v_min=v_min)),
              'RR': validate_rqa("%.3f", rp.recurrence_rate())}
  
  return rqa_vals[stat]

def validate_rqa(fmt, get_stat):
  # helper function 
  try: 
      return fmt % get_stat
  except IndexError: 
      return 'N/A'

def update_extent(im, data):
    new_lims = ((0, 0), (len(data), len(data)))
    im.axes.update_datalim(new_lims)
    im.axes.autoscale_view()
    im.axes.margins(x=0, y=0)
    im.set_extent([0, len(data), 0, len(data)])

# explicitly set width and height
layout = Layout(width='auto', height='25px', display="flex", margin="0px")

system_colors = {'White Noise': '#f561dd',
 'Sine': '#a48cf4',
 'Sinusoid': '#39a7d0',
 'Logistic Map': '#36ada4',
 'Brownian Motion': '#32b166'}


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

      :root {
        --bq-axis-tick-text-fill: black;
        --bq-axis-border-color: black;
        --bq-font: "serif", "sans-serif";
        --bq-axis-tick-text-font: "18px sans-serif";
        --bq-axislabel-font: 18px "sans-serif";
        --bq-mainheading-font: 18px "sans-serif"; 
        --bq-axis-path-stroke: black;
        # --bq-axis-tick-stroke: black;
      }

      .bqplot > svg .axis path {
      stroke: var(--bq-axis-path-stroke);
      }


    .bqplot .figure .jupyter-widgets .classic {
      font-family: "serif"
      }

      .plot-container .plotly .user-select-none .svg-container {
        width: 1100px;
        }
      
      #output-body {
          display: flex;
          align-items: center;
          justify-content: center;
          min-width: 100%;
          max-width: 100%;
          width: 100%;
      }

      .output_png {
          display: flex;
          align-items: center;
          justify-content: center;
      }

      .markdown blockquote {
        border-left: none !important; 
      }
      
        .jupyter-matplotlib-figure {
  height: 100%}
  .jupyter-widgets jupyter-matplotlib-canvas-div canvas {
  bottom: 0 !important;
  top: unset !important}

  .ipyevents-watched:focus {
    outline-width: '0' !important;
    outline: none !important;
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
    set_matplotlib_formats('svg')
  
    # Colab setup ------------------
    if "google.colab" in sys.modules:
        get_ipython().events.register('pre_run_cell', set_css)
        get_ipython().events.register('pre_run_cell', resize_colab_cell)
        display(Javascript("google.colab.output.resizeIframeToContent()"))
        display(Javascript('''google.colab.output.setIframeWidth(0, true, {maxWidth: 500})'''))
        fix_colab_outputs()

        # register custom widgets (bqplot)
        from google.colab import output
        output.enable_custom_widget_manager()
        

def add_colorbar(fig, mappable, axis, title):
    colorbar = fig.colorbar(mappable, ax=axis, pad=0.02)
    colorbar.ax.set_title(title, fontweight='demi', fontsize='xx-large')
    colorbar.ax.tick_params(which="minor", direction='out', length=3)
    colorbar.ax.tick_params(which="major", direction='out', length=8)
    colorbar.ax.set_autoscale_on(True)
    return colorbar
  
def create_slider(value, min, max, step, description="", label="", fmt=""):
    # Helper function to create a slider and its label
    slider_type = FloatSlider if "." in str(step) else IntSlider
    fmt = "%.2f" if "." in str(step) and not fmt else "%d" if "." not in str(step) and not fmt else "%s"
    slider = slider_type(value=value, min=min, max=max, step=step, description=description, continuous_update=False, readout=False, layout=slider_layout)
    label = Label(label if label else str(value))
    dlink((slider, "value"), (label, "value"), lambda x: fmt % x)
    return HBox([slider, label])

def create_text(value):
  label = Label(value)
  html = HTML()
  return HBox([label, html])

def create_col_center(*items):
  return VBox([*items], layout=Layout(display="flex", flex_direction="column", width="300px",
            align_self="center", justify_content="center", overflow="visible", align_items="flex-start"))
  
def create_row(*items):
  return HBox([*items], layout=Layout(display="flex", flex_direction="row", justify_content="space-between",
            align_items="center", overflow="visible", margin="12px 0 0 0"))
  
def create_row_stretch(*items):  
  return HBox([*items],
                    layout=Layout(width="1300px", display="flex", flex_direction="row", justify_items="center", align_items='stretch', overflow="visible")).add_class("row-stretch")

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
          
        elif get_ipython().__class__.__module__ != "google.colab._shell":  # check if Colab notebook
          return False
          
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
  
def setup_notebook():
  if in_notebook():
    display(HTML("""
      <style>
      
      div.prompt {
        min-width: fit-content: //80px;
      }
      
      .rendered_html h1 {}
      
      .prompt {
            text-align: left;
      font-size: 12px;
      }
      
      div.prompt:empty
      
      .text_cell_render .rendered_html > p:not(.fw) {
        align-self: center;
        width: 50% !important;
        max-width: 50% !important;
      }
      
      .rendered_html * + img {
        margin-top: 0;
      }
      
      .title .widget-html > .widget-html-content {
      align-self: unset;
      height: fit-content;
      }

      .title h2 {
          margin-top: 12px;
          margin-bottom: 0;
      }

      .flush-left {
      margin-left: 0 !important
      }

      .no-vert-margin {
      margin-top: 0 !important;
      margin-bottom: 0 !important;
      }

      .no-margin {
      margin: 0px !important;
      }

      :root {
      --jp-widgets-inline-label-width: 25px !important;
      --jp-widgets-margin: 5px !important;
      --jp-border-color1: black !important;
      --jp-widgets-dropdown-arrow: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMiIgZGF0YS1uYW1lPSJMYXllciAyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxOSAxMCI+CiAgPGRlZnM+CiAgICA8c3R5bGU+CiAgICAgIC5jbHMtMSB7CiAgICAgICAgc3Ryb2tlLXdpZHRoOiAwcHg7CiAgICAgIH0KICAgIDwvc3R5bGU+CiAgPC9kZWZzPgogIDxnIGlkPSJMYXllcl8xLTIiIGRhdGEtbmFtZT0iTGF5ZXIgMSI+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xOC4xNS4xNWMuMi0uMi41MS0uMi43MSwwLC4yLjIuMi41MSwwLC43MWwtOSw5Yy0uMi4yLS41MS4yLS43MSwwTC4xNS44NUMtLjA1LjY2LS4wNS4zNC4xNS4xNS4zNC0uMDUuNjYtLjA1Ljg1LjE1bDguNjUsOC42NUwxOC4xNS4xNVoiLz4KICA8L2c+Cjwvc3ZnPg==")}

      .toggle-button, .widget-toggle-button {
      width: fit-content !important;
      max-width: fit-content !important;
      margin: 5px !important;
      }

      .widget-toggle-buttons {
      height: fit-content !important;
      margin: 0 !important;
      }

      .pad-slider .slider-container {
      margin-left: 15px !important;
      }

      .widget-slider .noUi-handle, .jupyter-widget-slider .noUi-handle {
      border: none;
      top: calc((var(--jp-widgets-slider-track-thickness) - 12px) / 2);
      width: 12px;
      height: 12px;
      background: #0957b5;
      opacity: 0.95;
      }

      .widget-slider .noUi-handle:hover {
      background-color: black;
      border: 1px solid black;
      opacity: 1;
      }
      .widget-slider .noUi-connects {
      background: none;
      border: 1px solid black;
      }

      .jupyter-button.mod-active {
      background-color: #0957b5;
      color: white;
      opacity: 0.95;
      }

      .jupyter-button {
          border: 1px solid;
          height: 30px;
          background-color: white
      }

      .jupyter-widgets .widget-dropdown > select {
      border: 1px solid black;
      background-position: right 8px center;}

      </style>"""))
            
class RecurrenceMetrics:
  def __init__(self):
  
    features = {'Feature': ['Long diagonal lines',
                              'Diagonal lines of varying length',
                              'Short, chopped-up diagonal lines',
                              'Single, isolated points',
                              'Large, white patches',
                              'Vertical lines'],

                'Dynamical Behavior': ['Purely periodic motion',
                                      'Quasi-periodic motion',
                                      'Chaotic motion',
                                      'Stochastic (random) motion',
                                      'Disruption, possible state changes',
                                      'Trapped states, slowly changing dynamics']}
    metrics = {'RQA Metric': [
              '𝐷𝐸𝑇', '𝐿𝐴𝑀', '𝑅𝑅', '𝐿 𝑀𝐴𝑋', '𝐿 𝑀𝐸𝐴𝑁', '𝐿 𝐸𝑁𝑇𝑅', '𝐷𝐼𝑉', '𝑉 𝑀𝐴𝑋', '𝑇𝑇',  '𝑉 𝐸𝑁𝑇𝑅', '𝐶 𝐸𝑁𝑇𝑅', '𝑇 𝑅𝐸𝐶'],
              'Statistic': [
                  'Determinism', 'Laminarity', 'Recurrence Rate', 'Max Diagonal Line Length', 'Mean Diagonal Line Length',
                  'Diagonal Line Entropy', 'Divergence (1/𝐿 𝑀𝐴𝑋)', 'Max Vertical Line Length', 'Trapping Time', 'Vertical Line Entropy',
                  'Complexity Entropy', 'Mean Recurrence Time'
              ],

              'Definition': ['Fraction of recurrent points forming diagonal structures',
                            'Fraction of recurrent points forming vertical structures',
                            'Ratio of recurrent points to total points',
                            'Length of longest diagonal line',
                            'Average length of diagonal lines',
                            'Shannon Entropy of diagonal line length distribution',
                            'How quickly the diagonal line length distribution diverges',
                            'Length of longest vertical line',
                            'Average length of vertical lines',
                            'Shannon Entropy of vertical line length distribution',
                              'Level of intricacy or difficulty in understanding a system',
                              'Average time to return close to a previous state']
    }

    #RP Features table
    table1 = '<div id="features">'
    table1 += '<div style="text-align: center"><h2>RP Features</h2>\n'
    table1 += '<table>\n'
    table1 = f"{table1}<tr>{ ''.join(f'<th><h3>{s}</h3></th>' for s in features.keys()) }</tr>"
    for feature, behavior in zip(features['Feature'], features['Dynamical Behavior']):
        table1 = f"""{table1}<tr><td>{feature}</td><td>{behavior}</td></tr>"""
    table1 += "</table>"
    table1 += '</div>'


    # RQA metrics table
    table2 = '<div id="metrics">'
    table2 += '<div style="text-align: center"><h2>RQA Metrics</h2></div>\n'
    table2 += '<table>\n'
    metrics_cols = ['RQA Metric', 'Definition']
    table2 = f"""{table2}<tr>{ ''.join(f'<th colspan="2"><h3>{s}</h3></th>' if s == "RQA Metric" else f'<th><h3>{s}</h3></th>' for s in metrics_cols) }</tr>"""

    for metric, fullname, definition in zip(metrics['RQA Metric'], metrics['Statistic'], metrics['Definition']):
        if metric in ['𝐿 𝑀𝐴𝑋', '𝐿 𝑀𝐸𝐴𝑁']:
          table2 = f"""{table2}<tr><td>{metric}</td><td>{fullname}</td><td>{definition}<span> * </span></td></tr>"""
        else:
          table2 = f"""{table2}<tr><td>{metric}</td><td>{fullname}</td><td>{definition}</td></tr>"""
    table2 += "</table>"
    table2 += f"""<br/> <p><span> * </span> excluding the main diagonal (𝐿𝑂𝐼)
                and Theiler window (number of diagonals on either side of the 𝐿𝑂𝐼)</p>"""
    table2 += '</div>'
    
    display(HTML('''<style>

            #metrics table td:first-child {
              padding-right: 20px;
            }

            table tr:nth-child(odd) {
              background: var(--colab-secondary-surface-color);
              }

            table th {
              font-weight: bold;
              border-bottom: 1px solid black;
              vertical_align: middle;
              line-height: 1.3;
            }

            table th h3 {
              line-height: 1.3;
              transform: translateY(-25%);
            }

            table td {
              border: none;
              line-height: 1;
              overflow: hidden;
              padding: .5em;
              text-overflow: ellipsis;
              vertical-align: middle;
              white-space: nowrap;
              }

            table td > span, p span {
              color: red !important;
              font-size: 16px;
            }
            table td:last-child, th:last-child {
              border-left: 1px solid black;
            }

              table tr { display: table-row-group;
                vertical-align: middle;
                border-top-color: inherit;
                border-right-color: inherit;
                border-bottom-color: inherit;
                border-left-color: inherit;
                text-align: center;
                }

              table {border-collapse: collapse;
              border-spacing: 0;
              border: 1px solid black;
              font-size: 12px;
              margin-left: 8px;
              table-layout: fixed;
              }

              table thead {
                // border-bottom: 1px solid black;
                border-bottom: none;
                vertical-align: bottom;

                font-size: 14px;
                margin-bottom: 10px;
                display: block;
                margin:auto;
                text-align: center;
                }

            </style>'''))

    left_col = VBox([HTML(table1)], layout=Layout(display="flex", flex_direction="column", align_items="center", left="0", margin="0 auto", width="600px", flex_grow="1", flex_shrink="1"))
    right_col = VBox([HTML(table2)], layout=Layout(display="flex", flex_direction="column", align_items="center", right="0", margin="0 auto", width="800px", flex_grow="1", flex_shrink="1"))
    row = HBox([left_col, right_col], layout=Layout(display="flex", flex_direction="row", align_items="flex-start", width="100%"))
    
    display(row)