"""
Overview of dcp_client Package
==============================

The `dcp_client` package contains modules and subpackages for interacting with a server for model inference and training. It provides functionalities for managing GUI windows, handling image storage, and connecting to the server for model operations.

Subpackages
------------

- **dcp_client.gui package**: Contains modules for GUI components.
  
  - **Submodules**:
  
    - ``dcp_client.gui.main_window``: Defines the main application window and associated event functions.
    - ``dcp_client.gui.napari_window``: Manages the Napari window and its functionalities.
    - ``dcp_client.gui.welcome_window``: Implements the welcome window and its interactions.

- **dcp_client.utils package**: Contains utility modules for various tasks.
  
  - **Submodules**:
  
    - ``dcp_client.utils.bentoml_model``: Handles interactions with BentoML for model inference and training.
    - ``dcp_client.utils.fsimagestorage``: Provides functions for managing images stored in the filesystem.
    - ``dcp_client.utils.settings``: Defines initialization functions and settings.
    - ``dcp_client.utils.sync_src_dst``: Implements data synchronization between source and destination.
    - ``dcp_client.utils.utils``: Offers various utility functions for common tasks.

Submodules
------------

- **dcp_client.app module**: Defines the core application class and related functionalities.
  
  - **Classes**:
  
    - ``dcp_client.app.Application``: Represents the main application and provides methods for image management, model interaction, and server connectivity.
    - ``dcp_client.app.DataSync``: Abstract base class for data synchronization operations.
    - ``dcp_client.app.ImageStorage``: Abstract base class for image storage operations.
    - ``dcp_client.app.Model``: Abstract base class for model operations.

This package structure allows for easy management of GUI components, image storage, model interactions, and server connectivity within the dcp_client application.

"""