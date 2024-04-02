This repository contains various utility functions
for recurrence quantitative analysis (RQA) of time
series data as well as a tutorial notebook which
introduces recurrence analysis. The notebook
(`RQA_Tutorial.ipynb`) contains several interactive
widgets for exploring recurrence plots and metrics
for some characteristic dynamical systems can be
found in the repository's root directory.

Both the tutorial and source code make use of the package
[`pyunicorn`](https://github.com/pik-copan/pyunicorn)
for performing RQA. The code depends on an older version 
of `pyunicorn` which contains methods that were removed in
subsequent versions. Because this older version mysteriously
is no longer found on their repository, I've included this
specific version of `pyunicorn` via a [cached copy in my
repositories](https://github.com/keriheuer/pyunicorn) as a
project dependency. For the latest version of `pyunicorn`,
along with the software's license and README, please visit
their official repository.

The tutorial also requires this project (and LaTeX) to be
installed before it can be run. The notebook can be opened
in Google Colab if you'd like to run the notebook without
needing to install anything locally -- just make sure to
run the first cell of the notebook to download the necessary
dependencies in your Colab runtime.

In the case that you want to run the tutorial locally, please
ensure that you've installed this project before running the
notebook and that your local environment has a working LaTeX
compiler/distribution. If you don't have LaTeX (or don't want
to install it) and you run into `matplotlib` errors about
strings not getting compiled because the command `latex`
couldn't be found, try running the following lines before 
re-running any of the code cells in the tutorial notebook to
tell `matplotlib` not to use LaTeX for typesetting:
`import matplotlib as mpl
mpl.rcParams['text.usetex'] = False`

The tutorial notebook doesn't ship with the project's source code,
so if you'd like a copy of the notebook along with the RQA utilities,
you'll need to clone this repository to your local machine (i.e.,
`pip install git+https://github.com/keriheuer/rqa` will only install
the source code as the package `rqa`). For both the notebook and
source code, after using the command `git clone` on this repository,
simply `cd` into the cloned directory and run `pip install .` inside it.