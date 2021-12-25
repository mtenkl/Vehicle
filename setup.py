from setuptools import setup

setup(
   name='vehicle',
   version='1.1',
   description='A useful module',
   author='Michal Tenkl',
   author_email='',
   packages=['vehicle'],  #same as name
   install_requires=['numpy', 'matplotlib'], #external packages as dependencies
)