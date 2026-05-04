from setuptools import setup, find_packages

setup(
    name="fea_cfd_agent_system",
    version="1.0.0",
    description="Autonomous FEA & CFD ML Agent System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "langchain>=0.2.0",
        "langchain-anthropic>=0.1.0",
        "langgraph>=0.1.0",
        "pyvista>=0.43.0",
        "meshio>=5.3.0",
        "loguru>=0.7.2",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "mlflow>=2.10.0",
        "scikit-learn>=1.4.0",
        "requests>=2.31.0",
    ],
)
