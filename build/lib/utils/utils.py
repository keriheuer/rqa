from ipywidgets import Layout, HTML
from IPython.display import display, Javascript
from pyunicorn.timeseries import RecurrencePlot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt, numpy as np
import pkg_resources
# from . import package_name  # Import the package_name from your package's __init__.py
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Layout, HBox, Box, VBox, IntSlider, FloatSlider, Dropdown, ToggleButton, ToggleButtons, Label, HTML
from traitlets import dlink

col = Layout(display="flex", flex_direction="column", align_items="center", width="33.33%")
row = Layout(display="flex", flex_direction="row", width="100%", align_self="stretch", left="0", height="auto", text_align='left')
col_align_left = Layout(display="flex", flex_direction="row", width="100%", align_self="flex-start", left="0", height="60px", text_align='center', align_items="center")

widget_layout = Layout(width='100%', height='25px', margin="5px", font_family="serif")

button_layout = Layout(width='auto', max_width="250px", height='25px', margin="10px", font_family="serif")

ui_layout = Layout(display="flex", height="350px", flex_flow="column", overflow="visible",
    align_items="center", justify_content="center", width="45%", font_family="CMU Serif")

horizontal_layout = Layout(display="flex", flex_flow="row", height="350px", overflow= "visible",
    align_items="center", justify_content="center", font_family="serif")

vertical_layout = Layout(display="flex", flex_flow="column", height="350px", overflow= "visible",
    align_items="center", justify_content="center", font_family="serif")

label_layout = Layout(display="flex", flex_flow="row", align_items="center", justify_content="center",
    width="100%", text_align="center", height='40px', margin="10px", font_family="serif")

slider_layout = Layout(min_width="350px", font_family="serif")

dropdown_layout = Layout(max_width="200px", overflow="visible")

style = {'description_width': 'initial', "button_width": "auto", 'overflow':'visible',
            'white-space': 'nowrap', 'button_color': 'lemonchiffon', 'handle_color': 'cornflowerblue',
            'font_family': "serif"}

# explicitly set width and height
layout = Layout(width='auto', height='25px', display="flex", margin="10px", font_family="serif")

# function to get the full path of files in the package
def get_resource_path(relative_path):
    return pkg_resources.resource_filename('rqa', relative_path)

class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)
# For seamless integration we add the annotate3D method to the Axes3D class.

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)
    return annotation

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow


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

col = Layout(display="flex", flex_direction="column", align_items="center", width="33.33%")
row = Layout(display="flex", flex_direction="row", width="100%", align_self="stretch", left="0", height="auto", text_align='left')
col_align_left = Layout(display="flex", flex_direction="row", width="100%", align_self="flex-start", left="0", height="60px", text_align='center', align_items="center")

widget_layout = Layout(width='100%', height='25px', margin="5px")

button_layout = Layout(width='auto', max_width="fit-content", height='30px', margin="5px")

ui_layout = Layout(display="flex", height="350px", flex_direction="column", overflow="visible",
    align_items="center", justify_content="center", width="45%")

horizontal_layout = Layout(display="flex", flex_direction="row", height="350px", overflow= "visible",
    align_items="center", justify_content="center")

vertical_layout = Layout(display="flex", flex_direction="column", height="350px", overflow= "visible",
    align_items="center", justify_content="center")

label_layout = Layout(display="flex", flex_flow="row", align_items="center", justify_content="center",
    width="100%", text_align="center", height='40px', margin="0")

slider_layout = Layout(min_width="150px")

dropdown_layout = Layout(max_width="200px", overflow="visible")

style = {'description_width': 'initial', "button_width": "auto", 'overflow':'visible',
            'white-space': 'nowrap', 'button_color': 'lemonchiffon', 'handle_color': 'cornflowerblue'}

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
  from ipywidgets import Layout, HTML
  import ipywidgets as widgets
  display(widgets.HTML('''
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
            align_items="center", overflow="visible"))
  
def create_row_stretch(*items):  
  return HBox([*items],
                    layout=Layout(width="1300px", display="flex", flex_direction="row", justify_items="center", align_items='stretch', overflow="visible")).add_class("row-stretch")

def setup_notebook():
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
  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display
import ipyevents
# %matplotlib widget
from matplotlib.colors import to_rgb
# plt.ioff()

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