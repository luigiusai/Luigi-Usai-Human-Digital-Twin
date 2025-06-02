from setuptools import setup, find_packages

setup(
    name="digital-twin-software",
    version="0.1.0",
    author="Luigi Usai",
    author_email="usailuigi@gmail.com",
    description="Digital Twin Software con bias cognitivi e sistema di memoria avanzato",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/luigiusai/Luigi-Usai-Human-Digital-Twin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.13',
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dateutil>=2.8.2",
        "pydantic>=2.0.0",
        "pytest>=7.4.0",
        "black>=23.7.0",
        "isort>=5.12.0"
    ],
    package_data={
        "": ["*.json", "*.txt"]
    },
    include_package_data=True,
    zip_safe=False
)
