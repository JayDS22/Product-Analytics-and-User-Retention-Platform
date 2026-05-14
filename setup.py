from setuptools import setup, find_packages

setup(
    name="retention-platform",
    version="0.1.0",
    description="Product analytics and churn prediction platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "plotly>=5.18.0",
        "streamlit>=1.30.0",
        "pyarrow>=14.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
