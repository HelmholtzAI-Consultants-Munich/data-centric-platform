# Logging Configuration

This document describes the logging system implemented in the Data-Centric Platform (DCP).

## Overview

Both the client and server components now have comprehensive logging support:

- **Client**: `dcp_client/utils/logger.py`
- **Server**: `dcp_server/utils/logger.py`

Logs are written to both console (stdout) and optional log files with rotating file handler support.

## Client Logging

### Setup

The client logger is automatically initialized in `dcp_client/main.py`:

```python
from dcp_client.utils.logger import setup_logger, get_logger

# Initialize main logger with file output
setup_logger(log_file=os.path.join(os.path.expanduser("~"), ".dcp_client", "dcp_client.log"))

# Get logger for a module
logger = get_logger(__name__)
```

### Log File Location

By default, client logs are written to:
```
~/.dcp_client/dcp_client.log
```

The log file automatically rotates when it reaches 10MB, keeping up to 5 backup files.

### Log Levels

The default log level is `INFO`. To change it, modify the call in `main.py`:

```python
setup_logger(log_level=logging.DEBUG, log_file=...)  # for more verbose logging
setup_logger(log_level=logging.WARNING, log_file=...) # for less verbose logging
```

Available levels:
- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: Warning about something that should be noticed
- `ERROR`: A serious problem that should be addressed

## Server Logging

### Setup

The server logger is automatically initialized in `dcp_server/main.py`:

```python
from dcp_server.utils.logger import setup_logger, get_logger

# Initialize main logger with file output
setup_logger(log_file=os.path.join(os.path.expanduser("~"), ".dcp_server", "dcp_server.log"))

# Get logger for a module
logger = get_logger(__name__)
```

### Log File Location

By default, server logs are written to:
```
~/.dcp_server/dcp_server.log
```

The log file automatically rotates when it reaches 10MB, keeping up to 5 backup files.

## Using Logging in Your Code

### In Client Code

```python
from dcp_client.utils.logger import get_logger

logger = get_logger(__name__)

# Log messages
logger.debug("Detailed debug information")
logger.info("General information message")
logger.warning("Something that should be noticed")
logger.error("An error occurred")
logger.exception("An exception occurred (includes traceback)")
```

### In Server Code

```python
from dcp_server.utils.logger import get_logger

logger = get_logger(__name__)

# Log messages
logger.debug("Detailed debug information")
logger.info("General information message")
logger.warning("Something that should be noticed")
logger.error("An error occurred")
logger.exception("An exception occurred (includes traceback)")
```

## Log Format

### Console Output

```
2024-02-02 10:30:45 - dcp_client.main - INFO - Starting DCP Client...
2024-02-02 10:30:45 - dcp_client.utils.bentoml_model - DEBUG - Attempting to connect to BentoML server at http://0.0.0.0:7010
```

### File Output

The file output includes more detailed information:

```
2024-02-02 10:30:45 - dcp_client.main - INFO - [main.py:52] - Starting DCP Client...
2024-02-02 10:30:45 - dcp_client.utils.bentoml_model - DEBUG - [bentoml_model.py:25] - Attempting to connect to BentoML server at http://0.0.0.0:7010
```

## Current Logging Points

### Client

- **main.py**: Startup, mode selection, config loading, UI launch
- **bentoml_model.py**: Server connection attempts and inference operations

### Server

- **main.py**: Startup, configuration loading, BentoML server launch
- **service.py**: Configuration initialization, model setup, segment_image API calls

## Best Practices

1. **Use appropriate log levels**:
   - Use `DEBUG` for detailed information useful during development
   - Use `INFO` for key application events
   - Use `WARNING` for recoverable issues
   - Use `ERROR` for problems that prevent normal operation

2. **Include context**: Log relevant variables and parameters

   ```python
   logger.info(f"Processing image from {input_path}")
   ```

3. **Use exception logging**: When catching exceptions, use `logger.exception()` to include the traceback

   ```python
   try:
       # Some operation
   except Exception as e:
       logger.exception(f"Failed to process: {e}")
   ```

4. **Avoid logging sensitive data**: Do not log passwords, tokens, or other sensitive information

## Disabling File Logging

To log only to console (no file):

```python
setup_logger()  # No log_file parameter
```

## Advanced Configuration

To customize logging further, modify the `setup_logger()` function in:
- `dcp_client/utils/logger.py` (for client)
- `dcp_server/utils/logger.py` (for server)

For example, to add additional handlers or change formatting.
