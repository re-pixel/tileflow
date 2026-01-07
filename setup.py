from setuptools import find_packages, setup


setup(
	name="mini-compiler",
	version="0.0.0",
	description="Mini compiler: graph IR -> tiled uops -> simulated runtime",
	package_dir={"": "src"},
	packages=find_packages("src"),
	python_requires=">=3.10",
)
