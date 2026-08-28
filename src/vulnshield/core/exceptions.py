"""VulnShield-DNN custom exception hierarchy."""

class VulnShieldError(Exception):
    """Base exception for all VulnShield-DNN errors."""
    pass

class ConfigurationError(VulnShieldError):
    """Raised when configuration loading, parsing, or validation fails."""
    pass

class FaultInjectionError(VulnShieldError):
    """Raised when fault hook registration, execution, or removal fails."""
    pass

class ModelNotFoundError(VulnShieldError):
    """Raised when a requested model architecture or checkpoint is not found."""
    pass

class DatasetError(VulnShieldError):
    """Raised when dataset loading, splitting, or validation fails."""
    pass

class DiscoveryError(VulnShieldError):
    """Raised when RL discovery agent or environment encounters an error."""
    pass

class ProtectionError(VulnShieldError):
    """Raised when budget selection or fault-aware fine-tuning encounters an error."""
    pass

class EvaluationError(VulnShieldError):
    """Raised when evaluation metrics computation encounters an error."""
    pass
