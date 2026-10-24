"""Keyframe/common-generator composition plus protocol recovery intake."""
import functools, json, os
from pathlib import Path
from keyframe_binding import configure_inner as base_configure

def configure_inner(inner, selector_arm, row_root):
    base_configure(inner, selector_arm, row_root)
    from selector_prompt_binding import install_prompt
    install_prompt(inner, row_root)
    previous = inner.install_candidate_presentation

    @functools.wraps(previous)
    def install():
        value = previous()
        return value
    inner.install_candidate_presentation = install
