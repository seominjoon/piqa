#nsml: minjoon/research:180710

from distutils.core import setup

setup(
      name='minjoon/piqa',
      version='0.1',
      description='Phrase-Indexed Question Answering baseline models',
      install_requires =[
        "nltk",
        "numpy",
        "torch==0.4.1",
        "h5py",
    ]
)