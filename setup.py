from setuptools import find_packages
from setuptools import setup


setup(name='keras_text_summarization',
      version='0.0.1',
      description='Text Summarization in Keras using Seq2Seq and Recurrent Networks',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-text-summarization',
      download_url='https://github.com/chen0040/keras-text-summarization/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras'],
      packages=find_packages())
