"""Typed errors used across the project."""

class CarryOnError(Exception):
    """Base class."""

class PlannerVLMError(CarryOnError):
    pass

class PlanError(CarryOnError):
    pass

class EndpointNotFound(CarryOnError):
    pass

class TransportError(CarryOnError):
    pass

class CommandError(CarryOnError):
    pass

class UnresolvedPlaceholder(CarryOnError):
    pass
