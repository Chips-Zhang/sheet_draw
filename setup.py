from setuptools import setup, find_packages

setup(
    name="sheet-draw",
    version="0.1.0",
    description="乐谱绘制工具 - 将音频转换为简谱",
    author="chips",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "soundfile>=0.10.0",
        "av>=15; python_version >= '3.9'",
        "urllib3>=2,<3; python_version >= '3.9'",
        "yt-dlp>=2024.10.22; python_version >= '3.9'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sheet-draw=sheet_draw.cli:main",
        ],
    },
    python_requires=">=3.8",
)
