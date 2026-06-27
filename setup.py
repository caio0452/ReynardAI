from setuptools import setup, find_namespace_packages

setup(
    name="reynard_ai",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
)