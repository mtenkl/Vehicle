from setuptools import setup

setup(
   name='vehicle',
   version='1.3',
   description='A useful module',
   author='Michal Tenkl',
   author_email='',
   packages=['vehicle'],  #same as name
   install_requires=['numpy', 'matplotlib'], #external packages as dependencies
)