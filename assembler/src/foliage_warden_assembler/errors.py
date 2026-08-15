"""Assembler-specific fail-closed errors."""


class AssemblyError(ValueError):
    """Inputs cannot support the requested causal evidence assembly."""
