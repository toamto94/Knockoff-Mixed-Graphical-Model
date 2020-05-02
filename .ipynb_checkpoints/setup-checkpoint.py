from distutils.core import setup

setup(name='Ising_Knockoffs',
      version='0.0.2',
      description='Gibbs-Sampler, generating knockoffs for Ising distributed data',
      author='Tom Mueller',
      author_email='tom_mueller94@gmx.de',
      url='https://github.com/toamto94/Ising-Knockoffs',
      packages=['Ising_Knockoffs'],
      install_requires=['pandas==1.0.1', 'numpy==1.18.1', 'matplotlib==3.1.3']
     )
