# tests/test_postprocessors.py
import pytest
import torch
import numpy as np
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor
from beat_this.postprocessing_interface import Postprocessor

@pytest.fixture
def sample_logits():
    """Create sample logits for testing."""
    batch_size = 2
    seq_len = 100
    return (
        torch.randn(batch_size, seq_len),  # beat_logits
        torch.randn(batch_size, seq_len),  # downbeat_logits
        torch.ones(batch_size, seq_len, dtype=torch.bool)  # padding_mask
    )

def test_minimal_postprocessor_initialization():
    """Test MinimalPostprocessor initialization."""
    fps = 50
    processor = MinimalPostprocessor(fps=fps)
    assert processor.fps == fps

def test_minimal_postprocessor_call(sample_logits):
    """Test MinimalPostprocessor processing."""
    processor = MinimalPostprocessor(fps=50)
    beat_logits, downbeat_logits, padding_mask = sample_logits
    beat_times, downbeat_times = processor(beat_logits, downbeat_logits, padding_mask)
    assert isinstance(beat_times, list)
    assert isinstance(downbeat_times, list)
    assert all(isinstance(x, np.ndarray) for x in beat_times)
    assert all(isinstance(x, np.ndarray) for x in downbeat_times)

@pytest.mark.skip(reason="Requires madmom library which is not installed")
def test_dbn_postprocessor_initialization():
    """Test DbnPostprocessor initialization with different parameters."""
    # Test default parameters
    processor = DbnPostprocessor(fps=50)
    assert processor.fps == 50
    assert processor.beats_per_bar == (3, 4)
    assert processor.min_bpm == 55.0
    assert processor.max_bpm == 215.0
    assert processor.transition_lambda == 100

    # Test custom parameters
    custom_processor = DbnPostprocessor(
        fps=100,
        beats_per_bar=(4,),
        min_bpm=60.0,
        max_bpm=180.0,
        transition_lambda=50
    )
    assert custom_processor.fps == 100
    assert custom_processor.beats_per_bar == (4,)
    assert custom_processor.min_bpm == 60.0
    assert custom_processor.max_bpm == 180.0
    assert custom_processor.transition_lambda == 50

@pytest.mark.skip(reason="Requires madmom library which is not installed")
def test_dbn_postprocessor_call(sample_logits):
    """Test DbnPostprocessor processing."""
    processor = DbnPostprocessor(fps=50)
    beat_logits, downbeat_logits, padding_mask = sample_logits
    beat_times, downbeat_times = processor(beat_logits, downbeat_logits, padding_mask)
    assert isinstance(beat_times, list)
    assert isinstance(downbeat_times, list)
    assert all(isinstance(x, np.ndarray) for x in beat_times)
    assert all(isinstance(x, np.ndarray) for x in downbeat_times)

def test_postprocessor_interface_compliance():
    """Verify MinimalPostprocessor complies with the interface."""
    assert isinstance(MinimalPostprocessor(), Postprocessor)
    # Skip DbnPostprocessor check since it requires madmom

def test_minimal_postprocessor_with_synthetic_data():
    """Test MinimalPostprocessor with synthetic data that has clear peaks."""
    fps = 50
    processor = MinimalPostprocessor(fps=fps)
    
    # Create beat logits with peaks at frames 10, 30, 50
    beat_logits = torch.zeros(100)
    beat_logits[10] = 2.0
    beat_logits[30] = 2.0
    beat_logits[50] = 2.0
    
    # Create downbeat logits with a peak at frame 10
    downbeat_logits = torch.zeros(100)
    downbeat_logits[10] = 2.0
    
    # Process
    beat_times, downbeat_times = processor(beat_logits, downbeat_logits)
    
    # Check results
    assert len(beat_times) == 1  # Single item returned as a list with one element
    assert len(downbeat_times) == 1  # Single item returned as a list with one element
    
    # Access the arrays inside the lists
    beat_array = beat_times[0]
    downbeat_array = downbeat_times[0]
    
    # Check the expected values
    assert len(beat_array) == 3
    assert 10/fps in beat_array
    assert 30/fps in beat_array
    assert 50/fps in beat_array
    
    assert len(downbeat_array) == 1
    assert 10/fps in downbeat_array

def test_minimal_postprocessor_with_batched_data():
    """Test MinimalPostprocessor with batched data."""
    fps = 50
    processor = MinimalPostprocessor(fps=fps)
    
    # Create batched data
    batch_size = 2
    beat_logits = torch.zeros(batch_size, 100)
    downbeat_logits = torch.zeros(batch_size, 100)
    
    # First item: beats at 10, 30, 50 and downbeat at 10
    beat_logits[0, 10] = 2.0
    beat_logits[0, 30] = 2.0
    beat_logits[0, 50] = 2.0
    downbeat_logits[0, 10] = 2.0
    
    # Second item: beats at 20, 40, 60 and downbeat at 20
    beat_logits[1, 20] = 2.0
    beat_logits[1, 40] = 2.0
    beat_logits[1, 60] = 2.0
    downbeat_logits[1, 20] = 2.0
    
    # Process
    beat_times, downbeat_times = processor(beat_logits, downbeat_logits)
    
    # Check results
    assert len(beat_times) == batch_size
    assert len(downbeat_times) == batch_size
    
    # Check first item
    assert len(beat_times[0]) == 3
    assert len(downbeat_times[0]) == 1
    
    # Check second item
    assert len(beat_times[1]) == 3
    assert len(downbeat_times[1]) == 1

@pytest.mark.skip(reason="Requires madmom library which is not installed")
def test_dbn_postprocessor_with_synthetic_data():
    """Test DbnPostprocessor with synthetic data that has clear patterns."""
    fps = 50
    processor = DbnPostprocessor(fps=fps)
    
    # Create beat logits with a regular pattern (every 10 frames)
    beat_logits = torch.zeros(100)
    for i in range(0, 100, 10):
        beat_logits[i] = 2.0
    
    # Create downbeat logits with a regular pattern (every 40 frames)
    downbeat_logits = torch.zeros(100)
    for i in range(0, 100, 40):
        downbeat_logits[i] = 2.0
    
    # Process
    beat_times, downbeat_times = processor(beat_logits, downbeat_logits)
    
    # Check results - DBN might adjust the exact timing, but should find roughly the right pattern
    assert len(beat_times) == 1
    assert len(downbeat_times) == 1
    
    beat_array = beat_times[0]
    downbeat_array = downbeat_times[0]
    
    assert len(beat_array) > 0
    assert len(downbeat_array) > 0
    
    # Check that downbeats are a subset of beats
    for db in downbeat_array:
        assert any(np.isclose(db, bt, atol=0.1) for bt in beat_array) 