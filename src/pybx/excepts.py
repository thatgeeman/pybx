__all__ = ['NoIntersection', 'BxViolation']


class NoIntersection(Exception):
    """No intersection of boxes occur."""
    pass


class BxViolation(SyntaxWarning):
    """Violation of Bx properties."""
    pass
