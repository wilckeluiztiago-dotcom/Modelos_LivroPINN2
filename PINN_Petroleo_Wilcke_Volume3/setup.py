from setuptools import setup, find_packages

setup(
    name="pinn_petroleo_wilcke",
    version="3.0.0",
    author="Luiz Tiago Wilcke",
    author_email="contato@wilcke-petroleo.ai",
    description="Redes Neurais Informadas pela Física (PINNs) para Engenharia de Petróleo e Poços - Volume 3",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
    ],
    entry_points={
        "console_scripts": [
            "pinn-petroleo=pinn_petroleo_wilcke.main:main",
        ],
    },
)
