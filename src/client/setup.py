from setuptools import setup, find_packages
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation command to download SAM model checkpoints."""
    
    def run(self):
        install.run(self)
        self._download_sam_models()
    
    def _download_sam_models(self):
        """Download the appropriate SAM model checkpoint based on detected hardware."""
        try:
            from dcp_client.utils.sam_model_manager import SAMModelManager
            
            print("\nDetecting hardware and downloading appropriate SAM model...")
            manager = SAMModelManager()
            
            # Auto-detect hardware and select best model
            hardware = manager.detect_hardware()
            best_model = manager.select_model("auto")
            
            device_info = "CUDA GPU" if hardware["cuda"] else ("Apple Silicon (MPS)" if hardware["mps"] else "CPU")
            print(f"Detected: {device_info}")
            print(f"Selected model: {best_model}")
            
            try:
                print(f"Downloading {best_model} checkpoint...")
                manager.get_checkpoint_path(best_model)
                print(f"✓ {best_model} checkpoint downloaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Failed to download {best_model} checkpoint: {e}")
                print(f"  The checkpoint will be downloaded on first use.")
            
            print("SAM model checkpoint download complete.\n")
        except ImportError as e:
            print(f"⚠ Warning: Could not import SAMModelManager: {e}")
            print("  SAM model checkpoints will be downloaded on first use.\n")
        except Exception as e:
            print(f"⚠ Warning: Error during SAM model download: {e}")
            print("  SAM model checkpoints will be downloaded on first use.\n")


setup(
    name="data-centric-platform-client",
    version="0.2.0",
    description="The client of the data centric platform for microscopy image segmentation",
    author=["Christina Bukas", 
            "Helena Pelin",
            "Mariia Koren",
            "Marie Piraud"],
    author_email= ["christina.bukas@helmholtz-munich.de",
                   "helena.pelin@helmholtz-munich.de",
                   "mariia.koren@helmholtz-munich.de",
                   "marie.piraud@helmholtz-munich.de"],
    url="https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform",
    packages=find_packages(),
    include_package_data=True,
    package_data={"dcp_client": ["config.yaml", "config_remote.yaml"]},
    install_requires=[
        "matplotlib >=3.3",
        "scikit-image >=0.20",
        "scipy >=1.10", 
        "napari[pyqt5]>=0.4.17",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@main",
        "torch",
        "torchvision",
        "bentoml[grpc]>=1.2.5"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-qt>=4.2.0",
            "pytest-timeout",
            "pytest-mock",
            "sphinx",
            "sphinx-rtd-theme",
        ]
    },
    entry_points={
        "console_scripts": [
            "dcp-client=dcp_client.main:main",
        ]
    },
    cmdclass={
        "install": PostInstallCommand,
    },
    python_requires=">=3.9",
    keywords=[],  
    classifiers=[  
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md").read(),  
    maintainer=["Christina Bukas", "Helena Pelin"], 
    maintainer_email=["christina.bukas@helmholtz-munich.de", "helena.pelin@helmholtz-munich.de"],
    project_urls={ 
        "Repository": "https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform",
        "Documentation": "https://readthedocs.org/projects/data-centric-platform",
    }
)
