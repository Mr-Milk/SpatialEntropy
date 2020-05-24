from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="spatialentropy",
      packages=find_packages(),
      description="A python implementation of spatial entropy",
      long_description=long_description,
      long_description_content_type="text/markdown",
      version="0.0.2",
      author="Mr-Milk",
      author_email="zym.zym1220@gmail.com",
      url="https://github.com/Mr-Milk/SpatialEntropy",
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent",
          ],
      python_requires='>=3.5',
      install_requires=['numpy', 'scikit-learn'])
