from setuptools import find_packages, setup

setup(
    name="youtube-reddit_sentiment_analysis",
    version="0.1.0",
    author="Jeel Vaghasiya",
    description="An MLOps pipeline using a Stacking Classifier for YouTube & Reddit comment sentiment insights.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)