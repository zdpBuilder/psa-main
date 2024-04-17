from setuptools import setup, find_packages


# Package meta-data.
NAME = 'PSA'
DESCRIPTION = 'This package is written for PSA score prediction based on fine-tuned eres2net model'
URL = 'https://github.com/xxx/xxx'
EMAIL = 'zhangdaipeng@tju.edu.cn'
AUTHOR = 'zhangdaipeng'
REQUIRES_PYTHON = '>=3.7.7'
VERSION = '1.0'

setup(
    name="PSA",
    version="1.0",
    packages=find_packages(),
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=[
        'modelscope',
        'numpy',
        'torch>=1.7.1',
        'torchaudio>=0.7.2',
    ],
    include_package_data=True
)