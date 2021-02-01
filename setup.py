from setuptools import setup

setup(
    name="lifting-simulation-rl",
    version="0.0.1",
    install_requires=[
        "gym",
        "sympy",
        "scipy",
        "tensorflow==2.2.0",
        "keras==2.3.1",
        "hydra-core",
    ],
)
