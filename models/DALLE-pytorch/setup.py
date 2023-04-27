from setuptools import setup, find_packages

setup(
  name = 'dalle-pytorch',
  packages = find_packages(),
  include_package_data = True,
  version = '1.2.1',
  license='MIT',
  description = 'DALL-E - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/dalle-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'einops>=0.3.2',
    'ftfy',
    'g-mlp-pytorch',
    'pillow',
    'regex',
    'rotary-embedding-torch',
    'taming-transformers-rom1504',
    'tokenizers',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm',
    'youtokentome',
    'WebDataset'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
