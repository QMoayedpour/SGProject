from setuptools import setup, find_packages

setup(
    name="SGproject",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "seaborn", "numpy",
                      "matplotlib", "scikit-learn", "scipy",
                      "tqdm", "statsmodels", "xgboost", "catboost"],
)