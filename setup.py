"""
Setup script for the TradingAgents package.
"""

from setuptools import setup, find_namespace_packages

setup(
    name="tradingagents",
    version="0.1.0",
    description="Auditable multi-agent trading research framework for paper trading, strategy testing, and risk-controlled execution",
    author="TradingAgents Team",
    author_email="yijia.xiao@cs.ucla.edu",
    url="https://github.com/TauricResearch",
    packages=find_namespace_packages(include=["tradingagents*", "cli*", "webui*"]),
    include_package_data=True,
    package_data={
        "tradingagents.prompts": [
            "templates/*.md",
            "templates/*/*.md",
        ]
    },
    install_requires=[
        "openai>=2.33.0,<3.0.0",
        "langchain-core>=1.3.2,<2.0.0",
        "langchain-openai>=1.2.1,<2.0.0",
        "langchain-anthropic>=1.4.2,<2.0.0",
        "langchain-google-genai>=4.2.2,<5.0.0",
        "langchain-experimental>=0.4.1,<0.5.0",
        "langgraph>=1.1.10,<2.0.0",
        "langgraph-checkpoint-sqlite>=3.0.3,<4.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "praw>=7.7.0",
        "stockstats>=0.5.4",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "questionary>=2.0.1",
        "gradio>=4.0.0",
        "plotly>=5.18.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tradingagents=cli.main:app",
            "tradingagents-web=web_ui:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Trading Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
