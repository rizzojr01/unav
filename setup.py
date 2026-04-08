from setuptools import setup, find_packages

setup(
    name="unav",
    version="0.1.0",  # Update this version as you release new versions
    description="UNav: A Visual Navigation System",
    author="Anbang Yang",
    author_email="ay1620@nyu.edu",
    url="https://github.com/ai4ce/unav",  # Optional: Update with your project's repo
    packages=find_packages(),  # Automatically find all packages under the unav/ directory
    python_requires='>=3.8',  # Adjust this to match your project's requirements
    install_requires=[
        "dataloaders>=0.0.1",
        "einops>=0.8.1",
        "faiss-gpu>=1.7.2",
        "fast-pytorch-kmeans>=0.2.0.1",
        "h5py>=3.7.0",
        "joblib>=1.1.1",
        "kornia>=0.6.12",
        "lib>=4.0.0",
        "matplotlib>=3.7.1",
        "networkx>=2.8.4",
        "numpy>=1.22.4",
        "open3d>=0.19.0",
        "opencv-python>=4.10.0.84",
        # "opencv-python-headless>=4.10.0.84", # Uncomment if you only use headless version
        "Pillow>=10.0.0",
        "poselib>=2.0.0",
        "POT>=0.9.0",
        "prettytable<=3.11.0",
        "pyimplicitdist>=1.0.0",
        "pytorch-lightning>=2.0.6",
        "pytorch-metric-learning>=2.3.0",
        "PyYAML>=6.0",
        "scikit-image>=0.19.2",
        "scikit-learn>=1.2.1",
        "scipy>=1.10.0",
        "Shapely>=2.0.7",
        "timm>=0.4.12",
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "tqdm>=4.65.0",
        "transformers>=4.45.0",
        "tyro>=0.9.22",
        "wandb>=0.19.11",
        "xformers>=0.0.28",
    ],
    extras_require={
        "mast3r": [
            "mast3r @ git+https://github.com/naver/mast3r.git",
            "poselib>=2.0.0",
        ],
    },
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in or package_data
    package_data={
        "unav.core.third_party.SuperPoint_SuperGlue.extractors.SuperGluePretrainedNetwork.models.weights": ["*.pth"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",  # Update license as appropriate
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    # entry_points={
    #     "console_scripts": [
    #         "run_aligner=unav.run_aligner:main",  # Example: Expose CLI tool (implement main())
    #     ],
    # },
)
