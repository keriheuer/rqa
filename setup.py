from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:

                       setup(
     name="rqa101",
     version="0.0.1",
     url="https://github.com/keriheuer/rqa.git",
     author="Keri Heuer",
     license="MIT",
#     packages=["rqa", "rqa.utils", "rqa.timeseries", ],
     include_package_data=True,
     install_requires=["scipy==1.11.2", "ipywidgets", "pillow", "wheel", "numba==0.58.0",
                       "plotly", "bqplot", "matplotlib",
                       "setuptools>=65","Cython>=3.0.0","numpy>=1.24",
                       "pyunicorn @ git+https://github.com/pik-copan/pyunicorn.git#egg=pyunicorn"],
                       package_dir={'': 'src'},
     packages=find_packages(where="src"),
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
            'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7',
'Programming Language :: Python :: 3.8',
'Programming Language :: Python :: 3.9',
'Programming Language :: Python :: 3.10',
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]

  
)


from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="verysimplemodule", 
        version=VERSION,
        author="Jason Dsouza",
        author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
       )