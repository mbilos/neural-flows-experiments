from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'sklearn',
    'tqdm',
    'stribor==0.1.0',
    'torch>=1.8.0',
    'torchvision',
    'torchdiffeq',
    'reverse_geocoder',
    'lxml',
]

with open('README.md', 'r') as f:
    long_description = f.read()


setup(name='nfe',
      version='0.1.0',
      description='Neural flows experiments',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/mbilos/neural-flows-experiments',
      author='Marin Bilos',
      author_email='bilos@in.tum.de',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.6',
      zip_safe=False,
)
