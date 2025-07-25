from setuptools import setup, find_packages

setup(
    name="LLMGGen",
    version="0.1.0",
    description="A demo library",
    author="Alice",
    author_email="alice@example.com",
    license="MIT",
    python_requires=">=3.7",
    packages=[
        "LLMGGen",                 # 主包
        "LLMGGen.utils",           # 子模块 1
        "LLMGGen.eval_utils",      # 子模块 2
        "LLMGGen.jl_metric",       # 子模块 3
        "LLMGGen.models",          # 子模块 4
    ],
    package_dir={
        "LLMGGen": "src",                   # 主包路径
        "LLMGGen.utils": "src/utils",       # 子模块路径
        "LLMGGen.eval_utils": "src/eval_utils",
        "LLMGGen.jl_metric": "src/jl_metric",
        "LLMGGen.models": "src/models",
    },
)
