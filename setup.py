from setuptools import setup, find_packages

setup(
    name='sb3-rllab',
    version='0.024',
    packages=find_packages(include=['envtools.*']),
    url='https://github.com/cubecloud/sb3-rllab',
    license='MIT',
    author='cubecloud',
    author_email='zenroad60@gmail.com',
    description='Tools and extensions for stable baselines 3 RL package'
)
