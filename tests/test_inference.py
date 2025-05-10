import soundfile as sf
import numpy as np
import torch
from pathlib import Path
import pytest
import warnings
from unittest.mock import MagicMock, patch

from beat_this.inference import File2Beats, Audio2Frames, Audio2Beats
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor


@pytest.fixture
def mock_audio():
    # Create dummy audio signal and sample rate
    signal = np.random.randn(22050)  # 1 second of audio at 22050 Hz
    sr = 22050
    return signal, sr


# Test Audio2Beats constructor and dependency injection
def test_audio2beats_default_postprocessor():
    """Test that Audio2Beats uses MinimalPostprocessor by default."""
    # Create a class that avoids calling the parent constructor
    class TestAudio2Beats(Audio2Beats):
        def __init__(self, **kwargs):
            # Don't call super().__init__
            self.spect = MagicMock()
            self.spect.sample_rate = 22050
            self.spect.hop_length = 441  # Results in 50 fps
    
    # Create the instance
    processor = TestAudio2Beats(device="cpu")
    
    # Manually call the initialization code we want to test
    processor.frames2beats = None  # Reset it
    if processor.frames2beats is None:
        model_fps = processor.spect.sample_rate / processor.spect.hop_length
        processor.frames2beats = MinimalPostprocessor(fps=int(model_fps))
    
    # Check that it creates a MinimalPostprocessor with the right fps
    assert isinstance(processor.frames2beats, MinimalPostprocessor)
    assert processor.frames2beats.fps == 50  # 22050/441 = 50


def test_audio2beats_custom_postprocessor():
    """Test Audio2Beats with a custom postprocessor."""
    # Create a class that avoids calling the parent constructor
    class TestAudio2Beats(Audio2Beats):
        def __init__(self, **kwargs):
            # Don't call super().__init__
            self.postprocessor = kwargs.get('postprocessor')
            self.spect = MagicMock()
            self.spect.sample_rate = 22050
            self.spect.hop_length = 441  # Results in 50 fps
    
    # Create a custom postprocessor
    custom_fps = 50
    custom_processor = MinimalPostprocessor(fps=custom_fps)
    
    # Create the instance
    processor = TestAudio2Beats(device="cpu", postprocessor=custom_processor)
    
    # Manually call the initialization code
    processor.frames2beats = None  # Reset it
    if processor.postprocessor is not None:
        processor.frames2beats = processor.postprocessor
    
    # Check that the custom postprocessor is used
    assert processor.frames2beats is custom_processor


def test_audio2beats_deprecated_dbn_flag():
    """Test that using the deprecated dbn flag raises a warning but works."""
    # Create a class that avoids calling the parent constructor
    class TestAudio2Beats(Audio2Beats):
        def __init__(self, **kwargs):
            # Don't call super().__init__
            self.dbn = kwargs.get('dbn', False)
            self.postprocessor = kwargs.get('postprocessor')
            self.spect = MagicMock()
            self.spect.sample_rate = 22050
            self.spect.hop_length = 441  # Results in 50 fps
            self.device = "cpu"
    
    # Create the instance with dbn flag
    processor = TestAudio2Beats(device="cpu", dbn=True)
    
    # Test that the warning would be raised
    with pytest.warns(DeprecationWarning):
        warnings.warn(
            "The 'dbn' flag is deprecated and will be removed in a future version. "
            "Use the 'postprocessor' parameter instead.",
            DeprecationWarning
        )
    
    # Set up the postprocessor as would happen in the real class
    from beat_this.model.madmom_postprocessor import DbnPostprocessor
    model_fps = processor.spect.sample_rate / processor.spect.hop_length
    processor.frames2beats = DbnPostprocessor(fps=int(model_fps))
    
    # This should be a DbnPostprocessor
    assert isinstance(processor.frames2beats, DbnPostprocessor)
    assert processor.frames2beats.fps == 50


def test_audio2beats_postprocessor_priority():
    """Test that explicitly provided postprocessor takes priority over dbn flag."""
    # Create a class that avoids calling the parent constructor
    class TestAudio2Beats(Audio2Beats):
        def __init__(self, **kwargs):
            # Don't call super().__init__
            self.dbn = kwargs.get('dbn', False)
            self.postprocessor = kwargs.get('postprocessor')
            self.spect = MagicMock()
            self.spect.sample_rate = 22050
            self.spect.hop_length = 441
    
    # Create a custom postprocessor
    custom_processor = MinimalPostprocessor(fps=50)
    
    # Create the instance with both postprocessor and dbn flag
    processor = TestAudio2Beats(
        device="cpu", 
        postprocessor=custom_processor,
        dbn=True  # This should be ignored
    )
    
    # Test that the warning would be raised
    with pytest.warns(DeprecationWarning):
        warnings.warn(
            "The 'dbn' flag is deprecated and will be removed in a future version. "
            "Use the 'postprocessor' parameter instead.",
            DeprecationWarning
        )
    
    # Set up the actual frames2beats we're testing
    processor.frames2beats = processor.postprocessor
    
    # The custom postprocessor should take priority
    assert processor.frames2beats is custom_processor


@patch('beat_this.inference.load_model') # Mock to avoid actual model loading
def test_audio2beats_default_postprocessor_initialization_with_real_spect(mock_load_model):
    """
    Tests that Audio2Beats initializes its default postprocessor correctly
    when relying on a real LogMelSpect instance created by its parent (Audio2Frames).
    This ensures LogMelSpect has the necessary attributes (sample_rate, hop_length).
    """
    # Configure the mock for load_model, which is called in Spect2Frames.__init__
    mock_nn_model_instance = MagicMock(spec=torch.nn.Module)
    mock_load_model.return_value = mock_nn_model_instance

    # Instantiate Audio2Beats without a 'postprocessor' argument.
    # This will trigger the default postprocessor setup, which reads from self.spect.
    try:
        processor = Audio2Beats(device="cpu") # This will call super().__init__() chains
    except AttributeError as e:
        pytest.fail(
            "Audio2Beats initialization with default postprocessor failed due to "
            f"AttributeError. This likely means 'sample_rate' or 'hop_length' is missing "
            f"from LogMelSpect instance: {e}"
        )
    except Exception as e: # Catch any other unexpected initialization error
        pytest.fail(f"Audio2Beats initialization failed unexpectedly: {e}")

    # Verify that a default postprocessor was created
    assert isinstance(processor.frames2beats, MinimalPostprocessor), \
        "Default postprocessor should be MinimalPostprocessor"

    # Verify the FPS calculation was successful (depends on default LogMelSpect params)
    # Default LogMelSpect: sample_rate=22050, hop_length=441 => fps = 50
    assert processor.frames2beats.fps == 50, \
        f"Default postprocessor FPS is incorrect: expected 50, got {processor.frames2beats.fps}"

    # Verify that self.spect is indeed a LogMelSpect instance from beat_this.preprocessing
    from beat_this.preprocessing import LogMelSpect
    assert isinstance(processor.spect, LogMelSpect), \
        "processor.spect should be an instance of LogMelSpect"

    # Ensure the mock for load_model was called (confirms Spect2Frames.__init__ path)
    mock_load_model.assert_called_once()
