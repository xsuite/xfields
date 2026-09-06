# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #


def _resolve_filling_pattern(filling_pattern, filling_scheme):
    """Return the canonical filling input while accepting the legacy name."""
    if filling_pattern is not None and filling_scheme is not None:
        raise ValueError(
            'Only one of `filling_pattern` and `filling_scheme` can be '
            'provided.')
    if filling_pattern is not None:
        return filling_pattern
    return filling_scheme
