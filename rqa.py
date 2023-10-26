
def initialize_sliders():

    tau = widgets.IntSlider(value=1,min=1,max=10,step=1,description='Time Delay:',continuous_update=False, style=style, layout=widget_layout)# , readout=False) #,, height='50px'))

    dim = widgets.IntSlider(value=1,min=1,max=10,step=1,description='Embedding Dimension:',continuous_update=False, style=style, layout=widget_layout) #, readout=False) #,, height='50px'))

    normalize = widgets.ToggleButtons(options=["Yes", "No"], values=[True, False], description='Normalize?', disabled=False, style=style, layout=widget_layout) #,, height='50px'))

    metric = widgets.ToggleButtons(options=['Euclidean', 'Manhattan', 'Supremum'], values=["manhattan", "euclidean", "supremum"],
                        description='Distance Metric:',disabled=False, button_style='', style=style, layout=widget_layout)#, height='50px'))

    threshold = widgets.ToggleButtons(description="Threshold? ", options=['Local', 'Global'], style=style, layout=widget_layout) #,, height='50px'))

    RR = widgets.IntSlider(value=10,min=1,max=99,step=1,
        description='Recurrence Rate (%):',continuous_update=False, style=style, layout=widget_layout)#, readout=False) #,, height='50px'))

    plotting = widgets.ToggleButtons(options=['Thresholded', 'Unthresholded'],
        description='Plot which RP?',disabled=False, button_style='', style=style, layout=widget_layout) #,, height='50px'))

    signal = widgets.Dropdown(
        description='Time Series:   ',
        options=['Sine', 'Cosine', 'Sine + Sine', 'Sine + Cosine', 'Brownian Motion', 'Disrupted Brownian Motion', 'Rossler Attractor', 'Logistic Map'],
        values=['Sine', 'Cosine', 'Sine + Sine', 'Sine + Cosine', 'Brownian Motion', 'Disrupted Brownian Motion', 'Rossler Attractor', 'Logistic Map'],)

    lmin = widgets.IntSlider(value=2,min=2,max=10,step=1,description='L min:',continuous_update=False, style=style, layout=widget_layout)#, readout=False) #,, height='50px'))
    vmin = widgets.IntSlider(value=2,min=2,max=10,step=1,description='V min:',continuous_update=False, style=style, layout=widget_layout)# , readout=False) #,, height='50px'))

def get_ts(data, signal_type):

    """ Update the stored time series
    in the data object based on the current
    parameter values of the sliders."""

    if signal_type == 'sine':
      data.ts = sine(N.value, omega.value)
    elif signal_type =='cosine':
      data.ts = sine(N.value, omega.value)
    elif signal_type == 'white_noise':
      data.ts = white_noise(N.value, mean=mean.value, std=std.value, factor=750)
    #elif signal_type == 'brownian':
    #elif signal_type == 'disrupted_brownian':
    #elif signal_type == 'rossler':
    elif signal_type == 'logistic_map':
      data.ts = logistic_map(r.value,x.value,drift.value,N.value,M.value)


def get_RP(data, tau, dim, normalize, metric, threshold, RR, plotting, debug=False):

  """ Update the RP in the data object
  based on the current parameter values
  of the sliders as given by **kwargs."""

  # ensure stored time series is an array
  if isinstance(data.ts, list):
    data.ts = np.array(data.ts)

  # turn RR % into fraction
  RR = RR/100
  if threshold == 'Local':
    rp = RecurrencePlot(data.ts, tau=tau, dim=dim, normalize=normalize, metric=metric.lower(), local_recurrence_rate=RR, silence_level=2)

  else:
    rp = RecurrencePlot(data.ts, tau=tau, dim=dim, normalize=normalize, metric=metric.lower(), recurrence_rate=RR, silence_level=2)

  # update stored RP instance
  data.rp = rp

  # plot RP
  plt.figure(figsize=(3.8,3.8))
  plt.tick_params(which='both', axis='both', bottom=False, left=False, labelbottom=True, labelleft=True)
  plt.xlabel('t$_{rest}$ (arb. units)')
  plt.ylabel('t$_{rest}$ (arb. units)')
  plt.locator_params(nbins=8)

  # plot thresholded RP in single color
  if plotting == 'Thresholded':
    cmap = mpl.colormaps['Purples']
    im = plt.imshow(rp.recurrence_matrix(), origin='lower', cmap='Purples')

  # add colorbar for the unthresholded RP
  else:
    im = plt.imshow(rp._distance_matrix, origin='lower', cmap='jet')
    plt.colorbar(im, label='Threshold Distance $\epsilon$', cmap='jet', shrink=0.8)

  plt.show()

def rqa_stats(data, tau, dim, normalize, metric, threshold, RR, plotting, l_min, v_min, debug=False):

  """ Print out some RQA statistics for
    the current RP stored in the data object."""

  print("mean:", np.mean(data.ts))
  print("RQA Statistics\n---------------------")
  print("Determinism: %.2f" % data.rp.determinism(l_min))
  print("Laminarity: %.2f" % data.rp.laminarity(v_min))
  print("Recurrence Rate: %.2f" % data.rp.recurrence_rate())
  print("Trapping Time: %.2f" % data.rp.trapping_time(v_min))
  print("Diagonal Entropy: %.2f" % data.rp.diag_entropy(l_min))
  print('L max: %.2f' % data.rp.max_diaglength())
  print('L mean: %.2f' % data.rp.average_diaglength(l_min))
  print('Divergence (1/L) : %.2f' % (1/data.rp.max_diaglength()))
  print("Vertical Entropy: %.2f" % data.rp.vert_entropy(v_min))
  print('V max: %.2f' % data.rp.max_vertlength())
  print('V mean: %.2f' % data.rp.average_vertlength(v_min))

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

  # unmask any diagonals beyond theiler window
  # if new mask is smaller than old mask
  # if theiler < len(mask[mask == True]):
  #   unmask = np.zeros_like(rp.R, dtype=bool)

  #   for i in range(len(rp_matrix)):
  #     unmask[i:i+n-len(mask[mask == True]), i:i+n-len(mask[mask == True])] = False

  #   rp.R[unmask] = 1

  return mask

def include_diagonals(rp, mask):

  # clear pyunicorn cache for RQA
  rp.clear_cache(irreversible=True)

  # set masked diagonals back to 1
  rp.R[mask] = 1
