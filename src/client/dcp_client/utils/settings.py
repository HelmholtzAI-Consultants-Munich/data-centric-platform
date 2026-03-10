# Default values so attributes exist before init() is called (e.g. in tests).
accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
seg_name_string = "_seg"


def init() -> None:
    """Initialise global variables."""
    global accepted_types
    accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
    global seg_name_string
    seg_name_string = "_seg"
