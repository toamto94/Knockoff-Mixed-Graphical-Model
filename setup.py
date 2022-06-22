from distutils.core import setup

setup(name='KnockoffMixedGraphicalModel',
      version='0.0.2',
      description='Algorithm, training a mixed graphical model, using knockoffs',
      author='Tom Mueller',
      author_email='tom_mueller94@gmx.de',
      url='https://github.com/toamto94/Knockoff-Mixed-Graphical-Model',
      packages=['KnockoffMixedGraphicalModel'],
      install_requires=['pandas==1.0.1', 'numpy==1.22.0', 'scikit-learn==0.22.1']
     )
