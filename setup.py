from setuptools import setup, find_packages

setup(
    name="llmggen",
    version="0.1.0",
    description="A demo library",
    author="Alice",
    author_email="alice@example.com",
    license="MIT",
    python_requires=">=3.7",
    packages=[
        "llmggen",                 # 主包
        "llmggen.utils",           # 子模块 1
        "llmggen.eval_utils",      # 子模块 2
        "llmggen.jl_metric",       # 子模块 3
        "llmggen.models",          # 子模块 4
    ],
    package_dir={
        "llmggen": "src",                   # 主包路径
        "llmggen.utils": "src/utils",       # 子模块路径
        "llmggen.eval_utils": "src/eval_utils",
        "llmggen.jl_metric": "src/jl_metric",
        "llmggen.models": "src/models",
    },
)
