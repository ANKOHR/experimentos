class ExperimentOSError(Exception):
    """Base class for expected user-facing ExperimentOS errors."""


class NotFoundError(ExperimentOSError):
    pass


class InvalidConfigError(ExperimentOSError):
    pass


class UncitedFindingError(ExperimentOSError):
    """Raised when a finding is submitted without at least one run citation."""


class InvalidPredictionError(ExperimentOSError):
    pass
