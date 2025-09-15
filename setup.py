from setuptools import setup, find_packages

setup(
    name="rsna_aneurysm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "pydicom>=2.1.0",
        "scikit-learn>=0.24.0",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "tqdm>=4.56.0",
        "pyyaml>=5.4.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "rsna-train=rsna_aneurysm.train:main",
            "rsna-infer=rsna_aneurysm.infer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
)
