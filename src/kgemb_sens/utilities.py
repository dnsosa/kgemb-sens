"""General utilities for all code."""

# -*- coding: utf-8 -*-

import math


def good_round(x):
    frac = x - math.floor(x)
    if frac < 0.5:
        return math.floor(x)
    return math.ceil(x)
