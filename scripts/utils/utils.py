class DotDict(dict):
    """A dictionary that allows dot notation to access its values."""
    
    def __getattr__(self, key):
        """Return the value of the key as an attribute."""
        return self[key]
    
    def __setattr__(self, key, value):
        """Set the value of the key as an attribute."""
        self[key] = value