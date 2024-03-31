# rqa/config.py

import seaborn as sns
import numpy as np
import pandas as pd
from .utils import *

import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

import matplotlib as mpl
import matplotlib.font_manager as fm

font= fm.FontEntry(
    fname=get_resource_path("data/cmunrm.ttf"),
    name='cmunrm')
fm.fontManager.ttflist.insert(0, font)
mpl.rcParams['font.family'] = font.name

mpl.style.use(get_resource_path("data/standard.mplstyle"))

from traitlets import dlink
import numpy as np
from ipywidgets import Layout, HBox, Box, VBox, IntSlider, FloatSlider, Dropdown, ToggleButton, ToggleButtons, Label, HTML
import pandas as pd, sys
from pyunicorn.timeseries import RecurrencePlot
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap = sns.color_palette("hls", 8).as_hex()[::-1][:5]
systems = np.array(['White Noise', 'Sine', 'Sinusoid', 'Logistic Map', 'Brownian Motion'])
stats = np.array(['DET', 'LAM', 'L MAX', 'L MEAN', 'V MAX', 'V MEAN', 'L ENTR', 'V ENTR', 'DIV', 'TT'])

# pre-generated time series data
white_noise = np.load(get_resource_path("data/example_timeseries/white_noise_ts.npy"))
sine = np.load(get_resource_path("data/example_timeseries/sine_ts.npy"))
super_sine = np.load(get_resource_path("data/example_timeseries/super_sine_ts.npy"))
logi = np.load(get_resource_path("data/example_timeseries/logistic_map_ts.npy"))
brownian = np.load(get_resource_path("data/example_timeseries/brownian_ts.npy"))
signals_ts = [white_noise, sine, super_sine, logi, brownian]

# corresponding RPs
white_noise_rp = np.load(get_resource_path("data/example_rps/white_noise_rp.npy"))
sine_rp = np.load(get_resource_path("data/example_rps/sine_rp.npy"))
super_sine_rp = np.load(get_resource_path("data/example_rps/super_sine_rp.npy"))
logi_rp = np.load(get_resource_path("data/example_rps/logistic_map_rp.npy"))
brownian_rp = np.load(get_resource_path("data/example_rps/brownian_rp.npy"))
signals_rp = [white_noise_rp, sine_rp, super_sine_rp, logi_rp, brownian_rp]

colors, timeseries = {}, {}
characteristic_rqa_stats = {}
ts_data = [white_noise, sine, super_sine, logi, brownian]
for i, s in enumerate(systems):
    colors[s] = cmap[i]
    timeseries[s] = ts_data[i]
    characteristic_rqa_stats[s] = {}

df = pd.read_csv(get_resource_path('data/characteristic_systems_rqa_exclude_theiler.csv'))
for i, s in enumerate(systems):
    for stat, val in zip(df.columns, df.iloc[i].values):
        characteristic_rqa_stats[s][stat] = val
        
setup_notebook()
