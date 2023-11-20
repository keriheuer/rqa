from setuptools import setup, find_packages


VERSION = 'dev' 
DESCRIPTION = 'Introduction to Recurrence Analysis'
LONG_DESCRIPTION = 'An introduction to recurrence analysis that explores recurrence plots for standard dynamical systems and how they depend on various recurrence analysis parameters.'


with open('README.md', 'r', encoding='utf-8') as fh:

                       setup(
     name="rqa",
#     version="0.0.1",
    version="dev",
     url="https://github.com/keriheuer/rqa.git",
     author="Keri Heuer",
     license="MIT",
#     packages=["rqa", "rqa.utils", "rqa.timeseries", ],
     include_package_data=True,
     install_requires=["scipy==1.11.2", "ipywidgets", "pillow", "wheel", 
                       #"numba==0.58.0",
                       "plotly", "bqplot", "matplotlib",
                       "setuptools>=65","Cython>=3.0.0",
                       #"numpy>=1.24",
                #       "pyunicorn @ git+https://github.com/pik-copan/pyunicorn.git#egg=pyunicorn"],
#  package_dir={'': 'src'},
 #   packages=find_packages(where="src"),
                         packages=find_packages(),
    # py_modules=["thinkdsp"],
     python_requires='>=3',
  #   data_files=[('my_data', ['data/data_file'])],
  author_email="keri.heuer@drexel.edu",
  description="Code written in Python for an introduction to recurrence analysis.",
  long_description = fh.read(),
   keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
'Programming Language :: Python :: 3.9',
'Programming Language :: Python :: 3.10',
            "Operating System :: MacOS :: MacOS X",
        ]

  
)
