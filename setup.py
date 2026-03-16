from setuptools import setup, find_packages

setup(
    name='gpt',
    version='1.0.0',
    description='GPT2 implementation in PyTorch',
    author='Ashutosh Kothiwala',
    author_email='ashutosh@epifi.com',
    url='https://github.com/arkothiwala/gpt2',
    packages=['gpt'],
    python_requires='>=3.10',
    # install_requires=[
    #     'torch',
    #     'numpy',
    #     'pandas',
    #     # Add any other dependencies here
    # ]
)
