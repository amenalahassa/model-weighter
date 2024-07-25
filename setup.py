from setuptools import setup, find_packages

setup(
    name='model-weighter',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='Konrad Tagnon Amen ALAHASSA',
    author_email='am.konrad21@gmai.com',
    description='A package to estimate memory requirements for neural network models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/amenalahassa/model-weighter',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)