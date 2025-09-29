from setuptools import setup, find_packages

setup(
    name="Graphia",
    version="0.1.0",
    description="A demo library",
    author="Alice",
    author_email="alice@example.com",
    license="MIT",
    python_requires=">=3.7",
    packages=[
        "Graphia",                 # 主包
        "Graphia.utils",           # 子模块 1
        "Graphia.eval_utils",      # 子模块 2
        "Graphia.jl_metric",       # 子模块 3
        "Graphia.models",          # 子模块 4
    ],
    package_dir={
        "Graphia": "src",                   # 主包路径
        "Graphia.utils": "src/utils",       # 子模块路径
        "Graphia.eval_utils": "src/eval_utils",
        "Graphia.jl_metric": "src/jl_metric",
        "Graphia.models": "src/models",
    },
)
