from IPython.display import display, Javascript, HTML
from pyunicorn.timeseries import RecurrencePlot, CrossRecurrencePlot

from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output
from ipywidgets import IntSlider, FloatSlider, Dropdown, Layout, HBox, VBox
import sys, subprocess, os

def resize_colab_cell():
  display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 7000})'))
get_ipython().events.register('pre_run_cell', resize_colab_cell)

display(Javascript("google.colab.output.resizeIframeToContent()"))

display(Javascript('''google.colab.output.setIframeWidth(0, true, {maxWidth: 500})'''))

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)


display(HTML('''
<style>
  pre {
      white-space: normal;
  }
</style>
'''))

# Colab setup ------------------
if "google.colab" in sys.modules:
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

def style_output():
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



def style_plot(plot):
    plot.layout.width = "30%"
    plot.layout.flex_flow = "column"
    plot.layout.display = "flex"
    plot.layout.overflow = "visible"
    plot.layout.align_items = "center"
    plot.layout.justify_content = "center"
    plot.layout.height = "35%"

def style_rqa(rqa):
    rqa.layout.width = "25%"
    rqa.layout.display = "flex"
    rqa.layout.flex_flow="column"
    rqa.layout.overflow = "visible"
    rqa.layout.align_items = "center"
    rqa.layout.justify_content="center"

class TimeSeries():

  def __init__(self):
    self.ts = None
    self.rp = None

def update_params():

  data.tau = int(tau.value)
  data.dim = int(dim.value)
  data.normalize = normalize
  data.metric = metric.lower()
  data.threshold = threshold
  data.RR = int(RR.value)/100
  data.plotting = plotting

def initialize_ts_sliders():
   # widgets to set parameters of example time series

    duration = IntSlider(value=200,min=100,max=5000,step=50,description='Duration ',continuous_update=False)

    time_step = FloatSlider(value=0.01,min=-0.01,max=0.01,step=0.001,description='dt ',continuous_update=False)

    amplitude = FloatSlider(value=1,min=0.1,max=10,step=0.1,description='$A$ ',continuous_update=False,description_allow_html=True)

    frequency = IntSlider(value=300,min=50,max=1000,step=10,description='Frequency',continuous_update=False,description_allow_html=True)

    N = IntSlider(value=350,min=100,max=500,step=10,description='# Iterations ',continuous_update=False)

    y0 = IntSlider(value=5,min=1,max=30,step=1,description='y0 ',continuous_update=False)

    M = IntSlider(value=150,min=100,max=500,step=10,description='# RVS to use',continuous_update=False)

    # logistic map parameters
    r = FloatSlider(value=3, min=2,max=4,step=0.1,description='r ',continuous_update=False)

    x = FloatSlider(value=0.617,min=0,max=1,step=0.01,description='x ',continuous_update=False)

    drift = FloatSlider(value=0.0,min=-0.5,max=0.5,step=0.01,description='Drift ',continuous_update=False,description_allow_html=True)

    # periodic signals
    amp1 = FloatSlider(value=1,min=0.1,max=10,step=0.1,description='$\A_1$ ',continuous_update=False,description_allow_html=True)

    freq1 = IntSlider(value=300,min=50,max=1000,step=10,description='Frequency 1 ' ,continuous_update=False,description_allow_html=True)

    amp2 = FloatSlider(value=1,min=0.1,max=10,step=0.1,description='$\A_2$ ',continuous_update=False,description_allow_html=True)

    freq2 = IntSlider(value=300,min=50,max=1000,step=10,description='Frequency 2 ',continuous_update=False,description_allow_html=True)

    signal1_type = Dropdown(description='Signal 1 ',values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])

    signal2_type = Dropdown(description='Signal 2 ',values=['Sine', 'Cosine'],options=['Sine', 'Cosine'])

    mean = IntSlider(value=5,min=0,max=15,step=1,description='$\mu$ ',continuous_update=False,description_allow_html=True)

    std = FloatSlider(value=3,min=0.1,max=5,step=0.1,description='$\sigma$ ',continuous_update=False,description_allow_html=True)

    factor = IntSlider(value=300,min=100,max=1000,step=10,description='Amplification ',continuous_update=False,description_allow_html=True)

    # Rossler bifurcation parameter values
    a = FloatSlider(value=0.2,min=-0.5,max=0.55,step=0.01,description='a ',continuous_update=False,description_allow_html=True)

    b = FloatSlider(value=0.2,min=0,max=2,step=0.01,description='b ',continuous_update=False,description_allow_html=True)

    c = FloatSlider(value=5.7,min=0,max=45,step=0.01,description='c ',continuous_update=False,description_allow_html=True)

    signal1 = HBox(children=[signal1_type, freq1, amp1])
    signal2 = HBox(children=[signal2_type, freq2, amp2])

def initialize_rqa_sliders():
    # set up widgets for RQA

    tau = widgets.IntSlider(value=1,min=1,max=10,step=1,continuous_update=False, description="Time Delay (𝜏)", style=style, layout=slider_layout)

    dim = widgets.IntSlider(value=1,min=1,max=10,step=1,description='Embedding Dimension (𝑚):',continuous_update=False, style=style, layout=slider_layout)

    normalize = widgets.ToggleButton(description="Normalize time series", value=True, continuous_update=False, disabled=False, style=style, layout=button_layout)

    metric = widgets.ToggleButtons(options=['Euclidean', 'Manhattan', 'Supremum'], continuous_update=False, disabled=False, style=style, layout=layout)
    metric_label = widgets.HTML('Distance Metric:', layout=layout)

    RR = widgets.FloatSlider(value=0.1,min=0.01,max=0.99,step=0.01,description="Recurrence Rate (𝑅𝑅)", continuous_update=False, style=style, layout=slider_layout)

    which_RR = widgets.ToggleButtons(options=['Global 𝑅𝑅', 'Local 𝑅𝑅'], continuous_update=False, style=style, layout=layout)
    which_RR_label = widgets.HTML("For RQA, threshold by:", layout=layout)

    threshold = widgets.ToggleButton(description='Threshold RP', value=False, continuous_update=False, disabled=False, style=style, layout=button_layout)

    signal = widgets.Dropdown(
        description='Time Series:', options=['White Noise', 'Sine', 'Superimposed Sine', 'Logistic Map', 'Brownian Motion'],
        values=['White Noise', 'Sine', 'Cosine', 'Superimposed Sine', 'Logistic Map', 'Brownian Motion'])

    lmin = widgets.IntSlider(value=2,min=2,max=10,step=1,description='𝐿 𝑀𝐼𝑁:',continuous_update=False, style=style, layout=slider_layout)
    vmin = widgets.IntSlider(value=2,min=2,max=10,step=1,description='𝑉 𝑀𝐼𝑁:',continuous_update=False, style=style, layout=slider_layout)

    theiler = widgets.IntSlider(value=1, min=1, max=20, step=1, description="Theiler Window", continuous_update=False, style=style, layout=slider_layout)

    det = HBox([widgets.Label("Determinism:"), widgets.HTML()])
    lam = HBox([widgets.Label("Laminarity:"), widgets.HTML()])
    rr = HBox([widgets.Label("Recurrence Rate:"), widgets.HTML()])
    tt = HBox([widgets.Label("Trapping Time:"), widgets.HTML()])
    l_entr = HBox([widgets.Label("Diagonal Entropy:"), widgets.HTML()])
    l_max = HBox([widgets.Label("𝐿 𝑀𝐴𝑋:"), widgets.HTML()])
    l_mean = HBox([widgets.Label("𝐿 𝑀𝐸𝐴𝑁:"), widgets.HTML()])
    div = HBox([widgets.Label("Divergence (1 / 𝐿 𝑀𝐴𝑋):"), widgets.HTML()])
    v_entr = HBox([widgets.Label("Vertical Entropy:"), widgets.HTML()])
    v_max = HBox([widgets.Label("𝑉 𝑀𝐴𝑋:"), widgets.HTML()])
    v_mean = HBox([widgets.Label("𝑉 𝑀𝐸𝐴𝑁:"), widgets.HTML()])

    show_rqa_lines_label = widgets.HTML("Toggle to show:", layout=layout)
    show_theiler = widgets.ToggleButton(description='Theiler Window', value=False, continuous_update=False, disabled=False, style=style, layout=layout)
    show_lmax = widgets.ToggleButton(description='𝐿 𝑀𝐴𝑋', value=False, continuous_update=False, disabled=False, style=style, layout=layout)
    show_vmax = widgets.ToggleButton(description='𝑉 𝑀𝐴𝑋', value=False, continuous_update=False, disabled=False, style=style, layout=layout)



def set_style():
   # set up widget style and layout settings

    style = {'description_width': 'initial', "button_width": "auto", 'overflow':'visible' ,'white-space': 'nowrap', 'button_color': 'lemonchiffon', 'handle_color': 'cornflowerblue', 'font_family': "CMU Serif"}
    widget_layout = Layout(width='100%', height='25px', margin="5px", font_family="CMU Serif")
    button_layout = Layout(width='auto', height='25px', margin="10px", font_family="CMU Serif")
    ui_layout = Layout(display="flex", height="350px", flex_flow="column", overflow="visible", align_items="center", justify_content="center", width="45%", font_family="CMU Serif")
    horizontal_layout = Layout(display="flex", flex_flow="row", height="350px", overflow= "visible", align_items="center", justify_content="center", font_family="CMU Serif")
    vertical_layout = Layout(display="flex", flex_flow="column", height="350px", overflow= "visible", align_items="center", justify_content="center", font_family="CMU Serif")
    label_layout = Layout(display="flex", flex_flow="row", align_items="center", justify_content="center", width="100%", text_align="center", height='40px', margin="10px", font_family="CMU Serif")
    slider_layout = Layout(min_width="350px", font_family="CMU Serif")
    layout = Layout(width='auto', height='25px', display="flex", margin="10px", font_family="CMU Serif") #set width and height

def update_lightcurve(change):

    # update light curve
    ts = b.gen_random_walk(N.value)[:M.value]

    # if signal.value == 'Sine':
    #   ts = sine[:400]
    #   fig1.title = 'Sine Wave'
    #   distances = sine_rp._distance_matrix
    # elif signal.value == "Superimposed Sine":
    #   ts = super_sine[:300]
    #   fig1.title = 'Superimposed Sine Waves'
    #   distances = super_sine_rp._distance_matrix
    # elif signal.value == 'White Noise':
    #   ts = white_noise._distance_matrix
    #   fig1.title = 'White Noise'
    #   distances = white_noise_rp._distance_matrix
    # elif signal.value == "Logistic Map":
    #   ts = logi[:50]
    #   fig1.title = "Logistic Map"
    #   distances = logi_rp._distance_matrix
    # elif signal.value == "Brownian Motion":
    #   ts = dis[:300]
    #   fig1.title = "Brownian Motion"
    #   distances = logi_rp._distance_matrix

    with line.hold_sync():
      line.y = ts
      line.x = np.arange(len(ts))

def update_rp(change):

    # retrieve current time series
    ts = line.y

    brushing = window_selector.brushing
    if not brushing:
    # interval selector is active
      if line.selected is not None:
          # get the start and end indices of the interval
          start_ix, end_ix = line.selected[0], line.selected[-1]
      else:  # interval selector is *not* active
          start_ix, end_ix = 0, -1

    # update RP instance
    if which_RR.value.split()[0] == "Global":
      rp = RecurrencePlot(ts, normalize=normalize.value, metric=metric.value.lower(), recurrence_rate=RR.value, tau=tau.value, dim=dim.value, silence_level=2)
    else:
      rp = RecurrencePlot(ts, normalize=normalize.value, metric=metric.value.lower(), local_recurrence_rate=RR.value, tau=tau.value, dim=dim.value, silence_level=2)

    distances = rp._distance_matrix

   # update RP heatmap
    with rp_matrix.hold_sync():
        # fig2.title = title
        if not brushing:
          rp_matrix.x, rp_matrix.y = np.arange(len(distances))[start_ix:end_ix], np.arange(len(distances))[start_ix:end_ix]
          rp_matrix.color = distances[start_ix:end_ix]
        else:
          rp_matrix.x, rp_matrix.y = np.arange(len(distances)), np.arange(len(distances))
          rp_matrix.color = distances

    # update RQA
    l_min, v_min = lmin.value, vmin.value

    rr.children[1].value = "%.2f" % rp.recurrence_rate()
    lam.children[1].value = "%.2f" % rp.laminarity(v_min=v_min)
    v_entr.children[1].value = "%.2f" % rp.vert_entropy(v_min=v_min)
    tt.children[1].value = "%.2f" % rp.trapping_time(v_min=v_min)
    v_max.children[1].value = "%.2f" % rp.max_vertlength()
    v_mean.children[1].value = "%.2f" % rp.average_vertlength(v_min=v_min)

    # exclude diagonals
    mask = exclude_diagonals(rp, theiler.value)

    det.children[1].value = "%.2f" % rp.determinism(l_min=l_min)
    l_max.children[1].value = "%.2f" % rp.max_diaglength()
    l_mean.children[1].value = "%.2f" % rp.average_diaglength(l_min=l_min)
    div.children[1].value = "%.2f" % (1/rp.max_diaglength())
    l_entr.children[1].value = "%.2f" % rp.diag_entropy(l_min=l_min)

def update_rp2(change):

    # retrieve current time series
    ts = line.y

    # update RP instance
    if which_RR.value.split()[0] == "Global":
      rp = RecurrencePlot(ts, normalize=normalize.value, metric=metric.value.lower(), recurrence_rate=RR.value, tau=tau.value, dim=dim.value, silence_level=2)
    else:
      rp = RecurrencePlot(ts, normalize=normalize.value, metric=metric.value.lower(), local_recurrence_rate=RR.value, tau=tau.value, dim=dim.value, silence_level=2)

    # exclude Theiler window
    if threshold.value == True: # threshold the RP by recurrence rate
      distances = rp.R #recurrence_matrix()
      title = "Recurrence Plot"
      cs2.scheme = "Greys"

    else: # plot unthresholded recurrence plot (i.e. the distance matrix)
      distances = rp._distance_matrix
      title = "Distance Matrix"
      cs2.scheme = "plasma"

    # plot RP before excluding diagonals
    with rp_matrix2.hold_sync():
      # fig2.title = title
      rp_matrix2.x, rp_matrix2.y = np.arange(len(distances)), np.arange(len(distances))
      rp_matrix2.color = distances

    # update RQA
    l_min, v_min = lmin.value, vmin.value

    rr.children[1].value = "%.2f" % rp.recurrence_rate()
    lam.children[1].value = "%.2f" % rp.laminarity(v_min=v_min)
    v_entr.children[1].value = "%.2f" % rp.vert_entropy(v_min=v_min)
    tt.children[1].value = "%.2f" % rp.trapping_time(v_min=v_min)
    v_max.children[1].value = "%d" % rp.max_vertlength()
    v_mean.children[1].value = "%d" % rp.average_vertlength(v_min=v_min)

    # exclude diagonals
    mask = exclude_diagonals(rp, theiler.value)

    det.children[1].value = "%.2f" % rp.determinism(l_min=l_min)
    l_max.children[1].value = "%d" % rp.max_diaglength()
    l_mean.children[1].value = "%d" % rp.average_diaglength(l_min=l_min)
    div.children[1].value = "%d" % (1/rp.max_diaglength())
    l_entr.children[1].value = "%.2f" % rp.diag_entropy(l_min=l_min)

    update_rqa_lines(rp.recurrence_matrix())

def update_theiler(change):

  distances = rp_matrix2.color

  if threshold.value:
      with theiler_line.hold_sync():
          y1, y2 = np.arange(len(distances))-theiler.value, np.arange(len(distances))+theiler.value
          x_area, y_area = transform_to_area(np.arange(len(distances)), y1, y2)
          y_area[y_area < 0] = 0
          y_area[y_area > len(distances)] = len(distances)
          theiler_line.x = x_area
          theiler_line.y = y_area

          if show_theiler.value:
            theiler_line.fill_opacities = [0.3]*2
          else:
            theiler_line.fill_opacities = [0]*2

def update_theiler_slider(change):
  if tau.value != 1 and dim.value != 1:
    theiler.max = (tau.value * dim.value) + 10

def update_vlines(change):

  distances = rp_matrix2.color

  if threshold.value:

    # find positions
    # vertical line positions
    length = int(v_max.children[1].value)
    # return lists of arrays with vertical lines x and y positions
    idx = np.array([find_vmax(distances, i, length) for i in range(len(distances))], dtype=object)
    idx = idx[idx != None]
    v_x, v_y = [], []
    for (x,y) in idx:
      v_x.append(np.array([x]*length))
      v_y.append(np.arange(y, y+length))

    # update vertical lines mark
    with vlines.hold_sync():
      vlines.x = v_x
      vlines.y = v_y
      vlines.colors = ['green']*len(idx)

      if show_vmax.value:
        vlines.opacities = [1]*len(idx)
      else:
        vlines.opacities = [0]*len(idx)

def update_longest_diag(change):

  distances = rp_matrix2.color

  if threshold.value:

      # diagonal line position
    length = int(l_max.children[1].value)
    loc = find_lmax(distances, length)[0]
    l_x, l_y = np.arange(loc[0],loc[0]+length), np.arange(loc[1],loc[1]+length)

    # update diagonal lines mark
    with longest_diag.hold_sync():
      longest_diag.x = l_x
      longest_diag.y = l_y
      longest_diag.colors = ['blue']

      if show_lmax.value:
        longest_diag.opacities = [1]
      else:
        longest_diag.opacities = [0]

def find_vmax(distances, i, vmax):
  # find start positions of all vertical lines with length = V MAX
  col = distances[i, :]
  idx_pairs = np.where(np.diff(np.hstack(([False],col==1,[False]))))[0].reshape(-1,2)
  lengths = np.diff(idx_pairs,axis=1) # island lengths
  # if np.max(lengths) == vmax:
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

def transform_to_area(x_data, y1, y2):
    return np.append(x_data, x_data[::-1]), np.append(y1, y2[::-1])

def characteristic_rps():
    display(HTML("""
    <style>
    #output-body {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """))

    colors = sns.husl_palette().as_hex()

    import matplotlib.pyplot as pplot
    timeseries = [white_noise, sine, super_sine, logi, dis]
    fig, ax = pplot.subplots(nrows=2, ncols=5, figsize=(15,6), gridspec_kw={"height_ratios": [1,1]})

    for i, (ts, rp) in enumerate(zip(timeseries, signals)):

    ax[0,i].set_title(titles[i])
    if i == 2 or i == 4:
        ax[0, i].plot(ts[:300], linewidth=1, color=colors[i])
    elif i== 3:
        ax[0,i].plot(ts[:50], linewidth=1, color=colors[i])
    elif i == 1:
        ax[0,i].plot(ts[:500], linewidth=1, color=colors[i])
    elif i == 0:
        ax[0,i].plot(ts[:400], linewidth=1, color=colors[i])
    ax[0,i].tick_params(which="both", axis="both", bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False, labeltop=False)

    ax[1,i].imshow(rp.recurrence_matrix(), cmap='Greys', origin='lower', interpolation='none')
    ax[1,i].tick_params(which="both", axis="both", bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False, labeltop=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.02)
    pplot.show()

def characteristic_rqa():
    layout = {'yaxis': {'range': [0, df['DET'].max()*1.1]}}
    figure = go.FigureWidget(data=[go.Bar(
        x=['White Noise', 'Sine', 'Superimposed Sine','Logistic Map', 'Brownian Motion'],
        y=df['DET'].values,
        marker_color=colors, # marker color can be a single color value or an iterable
        text=df['DET'].values,
        textposition="outside",
        texttemplate="%{y:.2%}",
        hoverinfo='skip'
    )]) #, layout_yaxis_range=[0,50])

    # Update plot sizing
    figure.update_layout(
        width=700,
        height=500,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
        hovermode=False,
        yaxis_title='Fraction of Recurrences'
        # config = {'displayModeBar': False}
    )

    bar = figure.data[0]

    stats = np.array(['DET', 'LAM', 'L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'L ENTR', 'V ENTR', 'DIV', 'TT'])
    rqa_stat = widgets.Dropdown(options=stats)

    yvals = np.array([[5, 25, 15, 17, 30], [15, 2, 20, 8, 10], [30, 16, 5, 20, 7]])

    figure.observe(update_stat)
    rqa_stat.observe(update_stat)
    # figure.show(config = {'displayModeBar': False})

    display(VBox([rqa_stat, figure]))

def update_stat(change):
    """ Helper function to update statistics in characteristic RQA plot.
    """
    labels = np.array(['Determinism', 'Laminarity', 'Longest Diagonal Line Length', 'Average Diagonal Line Length', 'Longest Vertical Line Length', 'Average Vertical Line Length', 'Diagonal Line Entropy', 'Vertical Line Entropy', 'Divergence (1/L MAX)', 'Trapping Time'])
    i = np.where(rqa_stat.value == stats)[0][0]
    vals = df[rqa_stat.value].values
    bar.y = vals
    bar.text = vals

    if rqa_stat.value in ['L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'TT']:
        bar.texttemplate="%{y:d}" # integer
        ylabel = 'Time (arb. units)'
    elif rqa_stat.value == 'DIV':
        bar.texttemplate="%{y:.2f}" # integer
        ylabel = '1 / Time (arb. units)'
    elif 'ENTR' in stats[i]:
        bar.texttemplate="%{y:.2f}" # 2 decimal float, unitless
        ylabel = ""
    else:
        bar.texttemplate="%{y:.2%}" # 2 decimal percentage
        ylabel = 'Fraction of Recurrences'

    figure.for_each_yaxis(lambda x: x.update({'range': [0, df[rqa_stat.value].max() * 1.1]}))
    figure.update_layout(title_text=labels[i], yaxis_title=ylabel)

def generate_rp():
   ui = widgets.VBox([widgets.HTML("<h2>Brownian Motion</h2>"), N,M], layout=Layout(display="flex", overflow="visible"), style=style)

    # create a figure widget
    b = Brownian()
    ts = b.gen_random_walk(N.value)[:M.value]
    # ts = white_noise
    x_scale, y_scale = LinearScale(), LinearScale()
    xax = Axis(label="Time (arb. units)",scale=x_scale)
    yax = Axis(label="Amplitude",scale=LinearScale(),orientation="vertical") #, offset={'scale':y_scale, 'value':10})
    line = Lines(x=np.arange(len(ts)), y=ts, scales={"x": x_scale, "y": y_scale}, stroke_width=1)
    fig1 = Figure(marks=[line], axes=[xax, yax], title='Time Series', layout=Layout(height="400px", width="500px", display="flex", overflow="visible"), padding_y=0, style=style)
    # fig1 = Figure(marks=[line], axes=[xax, yax], title='Time Series', layout=Layout(height="300px", width="800px", display="flex", overflow="visible"), padding_y=0, style=style)
    fig1.layout.margin_right = "50px"

    # RP
    cs = ColorScale(scheme="plasma")
    y_scale2 = LinearScale()
    rp = RecurrencePlot(np.array(ts), normalize=False, metric='euclidean', recurrence_rate=0.10, silence_level=2) #, dim=2, tau=2)
    rp_matrix = HeatMap(color=rp._distance_matrix, scales={'color': cs})#  ,  'orientation': "vertical", "side": "right"})
    # rp_matrix = HeatMap(color=white_noise_rp._distance_matrix, scales={'color': cs})#  ,  'orientation': "vertical", "side": "right"})

    cax = ColorAxis(scale=cs, label=r"𝜀", orientation="vertical", side="right")
    yax2 = Axis(label="Time (arb. units)",scale=x_scale,orientation="vertical")

    fig2 = Figure(
        marks=[rp_matrix], #xlabel, ylabel, cblabel,
        axes=[cax, xax, yax2],
        title="Distance Matrix",
        layout=Layout(width="500px", height="500px", overflow="visible"),
        min_aspect_ratio=1,
        padding_y=0,
        style=style
    )

    plt.grids(value="none")

    data = TimeSeries()

    # 2. register the callback by using the 'observe' method
    N.observe(update_lightcurve) #, names = "value", type = "change")
    M.observe(update_lightcurve) #, names = "value", type = "change")
    # signal.observe(update_lightcurve)

    # register rqa callbacks
    line.observe(update_rp)

    from bqplot.interacts import BrushIntervalSelector
    window_selector = BrushIntervalSelector(marks=[line], scale=line.scales["x"])
    fig1.interaction = window_selector  # set the interval selector on the figure

    # register the callback with brushing trait of interval selector
    window_selector.observe(update_rp, "brushing")
    line.observe(update_rp, "brushing")

    # render the figure
    lightcurve = VBox([ui, fig1], layout=Layout(display="flex", width="60%", height="auto", overflow="visible", flex_flow="column", justify_content="space-between", align_items="flex-start"))
    display(widgets.HBox([lightcurve, fig2], layout=Layout(display="flex", height="auto", width="1000px", overflow="visible", flex_flow="row", justify_content="space-between", align_items="flex-end"))) #, layout=horizontal_layout))
    # lightcurve = HBox([ui, fig1], layout=Layout(display="flex", width="100%", height="auto", overflow="visible", flex_flow="row", justify_content="space-between", align_items="flex-start"))
    # display(widgets.VBox([lightcurve, HBox([rqa_ui, fig2, rqa_stats], layout=Layout(display="flex", flex_flow="row", align_items="center", width="100%", overflow='visible'))], layout=Layout(display="flex", height="auto", width="100%", overflow="visible", flex_flow="column", justify_content="space-between", align_items="flex-end"))) #, layout=horizontal_layout))

def vary_rqa_params():
   # RP

    line.observe(update_rp2)

    cs2 = ColorScale(scheme="plasma")

    x_sc = LinearScale(allow_padding=False)
    y_sc = LinearScale(allow_padding=False)

    c_ax = ColorAxis(scale=cs2, label=r"𝜀", orientation="vertical", side="right")
    x_ax = Axis(label="Time (arb. units)", scale=x_sc)
    y_ax = Axis(label="Amplitude", scale=y_sc,orientation="vertical")

    # rp = RecurrencePlot(line.y, normalize=False, metric='euclidean', recurrence_rate=0.10, silence_level=2, dim=2, tau=2)
    domain = np.arange(len(fig2.marks[0].color))
    dist = fig2.marks[0].color
    rp_matrix2 = HeatMap(color=dist, x=domain, y=domain, scales={"x": x_sc, "y": y_sc, 'color': cs2})#  ,  'orientation': "vertical", "side": "right"})
    theiler_line = Lines(x=domain, y=[domain, domain], scales={'x': x_sc, 'y': y_sc}, fill_colors='red', fill="inside", fill_opacities=[0, 0], stroke_width=0, apply_clip=True, preserve_domain={'x': True, 'y':True})

    vlines = Lines(x=domain, y=domain, scales={"x": x_scale, "y": y_scale}, stroke_width=1, colors=['green'], opacities=[0])
    longest_diag = Lines(x=domain, y=domain, scales={"x": x_scale, "y": y_scale}, stroke_width=1, colors=['blue'], opacities=[0])

    fig3 = Figure(
        marks=[rp_matrix2, theiler_line, vlines, longest_diag],
        axes=[c_ax, x_ax, y_ax],
        title="Distance Matrix",
        layout=Layout(width="500px", height="500px", overflow="visible"),
        min_aspect_ratio=1, max_aspect_ratio=1,
        padding_y=0, padding_x=0,
        style=style)

    plt.grids(value="none")

    label1 = widgets.HTML(value="<h3>Embedding Parameters</h3>", layout=label_layout)
    label2 = widgets.HTML(value="<h3>RP Parameters</h3>", layout=label_layout)
    label3 = widgets.HTML(value="<h3>RQA Parameters</h3>", layout=label_layout)
    rqa_ui = widgets.VBox([label1, VBox([tau, dim]), Box([normalize]), label2, HBox([metric_label, metric]),
                        RR, HBox([which_RR_label, which_RR]), Box([threshold]), label3, VBox([theiler, lmin, vmin])],
                        layout=Layout(width="500px", overflow="visible", margin_right="50px"))

    statistics = VBox([det, lam, rr, tt, v_entr, l_max, l_mean, div, v_entr, v_max, v_mean],
                    layout=Layout(display="flex", flex_direction="column", width="300px",
                                    align_self="center", justify_content="center", overflow="visible"))
    rqa_stats = widgets.Output()
    with rqa_stats:
    display(statistics, layout=Layout(display="flex", flex_direction="column", align_self="center"))
    # display(HBox([statistics], layout=Layout(overflow="visible", flex_direction="column", width="350px", margin_left="50px", align_self="center", justify_items="center")), layout=Layout(display="flex", flex_direction="column", align_self="center"))

    # register callbacks
    tau.observe(update_rp2)
    dim.observe(update_rp2)
    normalize.observe(update_rp2)
    metric.observe(update_rp2)
    threshold.observe(update_rp2)
    RR.observe(update_rp2)
    which_RR.observe(update_rp2)
    lmin.observe(update_rp2)
    vmin.observe(update_rp2)

    tau.observe(update_theiler_slider)
    dim.observe(update_theiler_slider)
    theiler.observe(update_theiler)
    theiler.observe(update_rp2)
    show_theiler.observe(update_theiler)

    l_max.observe(update_longest_diag)
    l_max.observe(update_rp2)
    threshold.observe(update_longest_diag)
    show_lmax.observe(update_longest_diag)

    v_max.observe(update_vlines)
    v_max.observe(update_rp2)
    threshold.observe(update_vlines)
    show_vmax.observe(update_vlines)

    fig3.layout.margin="0px 50px"
    show_rqa_lines = HBox([show_rqa_lines_label, show_theiler, show_lmax, show_vmax], style=style, layout=Layout(width="500px", height="30px", overflow="visible", flex="1", display="flex", align_self="center", align_items="center", flex_direction="row", justify_content="center"))
    # VBox([fig3, show_rqa_lines], style=style, layout=Layout(display="flex", flex_flow="column", justify_items="center", align_items="center"))
    display(HBox([rqa_ui, VBox([fig3, show_rqa_lines], layout=Layout(display="flex", overflow="visible", align_items="center", flex_direction="column", justify_items="center")),
                rqa_stats], layout=Layout(display="flex", flex_direction="row", justify_content="space-between", align_items="center", overflow="visible")))