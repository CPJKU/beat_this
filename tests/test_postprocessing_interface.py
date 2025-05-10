# tests/test_postprocessing_interface.py
import pytest
from beat_this.postprocessing_interface import Postprocessor
from typing import Protocol

def test_postprocessor_is_protocol():
    """Verify Postprocessor is a Protocol."""
    assert issubclass(Postprocessor, Protocol)

def test_postprocessor_method_signature():
    """Verify Postprocessor has the correct method signature."""
    import inspect
    sig = inspect.signature(Postprocessor.__call__)
    assert 'beat_logits' in sig.parameters
    assert 'downbeat_logits' in sig.parameters
    assert 'padding_mask' in sig.parameters
    assert sig.parameters['padding_mask'].default is None 