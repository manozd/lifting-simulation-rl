from setuptools import setup

setup(
    name="lifting-simulation-rl",
    version="0.0.1",
    install_requires=[
        "gym",
        "sympy",
        "scipy",
        "keras-rl",
        "keras==2.2.4",
        "tensorflow==1.13.1",
    ],
)
