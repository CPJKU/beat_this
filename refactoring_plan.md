# Refactoring Plan: Postprocessor Dependency Injection

This plan outlines the necessary changes to integrate Dependency Injection for the postprocessor component in the beat_this library, enhancing flexibility and modularity.

## Overview

The refactoring will be implemented in 6 main steps:
1. Define the Postprocessor Interface
2. Implement Concrete Postprocessor Classes
3. Inject Postprocessor into Inference Classes
4. Update the Command-Line Interface
5. Update the PyTorch Lightning Module
6. Update Launch Scripts

## Step 1: Define the Postprocessor Interface

### Goal
Establish a clear contract for all postprocessor implementations.

### Files Affected
- `beat_this/postprocessing_interface.py` (New File)

### Changes
Create a new file `beat_this/postprocessing_interface.py` with the following implementation:

```python
# beat_this/postprocessing_interface.py
import numpy as np
import torch
from typing import Protocol, Tuple, List

class Postprocessor(Protocol):
    """Interface for beat and downbeat postprocessors."""
    def __call__(
        self,
        beat_logits: torch.Tensor,
        downbeat_logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ...
```

### Testing
1. Create test file `tests/test_postprocessing_interface.py`:
```python
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
```

## Step 2: Implement Concrete Postprocessor Classes

### Goal
Create distinct classes for the "minimal" and "dbn" postprocessing logic, making DBN parameters configurable and ensuring both classes adhere to the Postprocessor interface.

### Files Affected
- `beat_this/model/postprocessor.py` (for MinimalPostprocessor)
- `beat_this/model/madmom_postprocessor.py` (New File, for DbnPostprocessor)

### Changes
```python
# beat_this/model/postprocessor.py
from beat_this.postprocessing_interface import Postprocessor as PostprocessorInterface

class MinimalPostprocessor(PostprocessorInterface):
    def __init__(self, fps: int = 50):
        self.fps = fps
    
    def __call__(self, beat_logits, downbeat_logits, padding_mask=None):
        # Implementation of minimal postprocessing logic
        pass

# Keep deduplicate_peaks utility function
```

```python
# beat_this/model/madmom_postprocessor.py
from beat_this.postprocessing_interface import Postprocessor as PostprocessorInterface
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

class DbnPostprocessor(PostprocessorInterface):
    def __init__(
        self, 
        fps: int = 50, 
        beats_per_bar=(3, 4), 
        min_bpm=55.0, 
        max_bpm=215.0, 
        transition_lambda=100
    ):
        self.fps = fps
        self.beats_per_bar = beats_per_bar
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.transition_lambda = transition_lambda
        # Initialize madmom DBN
        self.dbn = DBNDownBeatTrackingProcessor(
            fps=self.fps,
            beats_per_bar=self.beats_per_bar,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            transition_lambda=self.transition_lambda
        )
    
    def __call__(self, beat_logits, downbeat_logits, padding_mask=None):
        # Implementation of DBN postprocessing logic using self.dbn
        pass
```

### Testing
1. Create test file `tests/test_postprocessors.py`:
```python
# tests/test_postprocessors.py
import pytest
import torch
import numpy as np
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor

@pytest.fixture
def sample_logits():
    """Create sample logits for testing."""
    return (
        torch.randn(100, 2),  # beat_logits
        torch.randn(100, 2),  # downbeat_logits
        torch.ones(100, dtype=torch.bool)  # padding_mask
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
    """Verify both postprocessors comply with the interface."""
    from beat_this.postprocessing_interface import Postprocessor
    assert issubclass(MinimalPostprocessor, Postprocessor)
    assert issubclass(DbnPostprocessor, Postprocessor)
```

## Step 3: Inject Postprocessor into Inference Classes

### Goal
Modify the inference pipeline classes to accept an instance of the Postprocessor interface during initialization.

### Files Affected
- `beat_this/inference.py`

### Changes
```python
# beat_this/inference.py
from beat_this.postprocessing_interface import Postprocessor as PostprocessorInterface
from beat_this.model.postprocessor import MinimalPostprocessor

class Audio2Beats(Audio2Frames):
    def __init__(
        self,
        checkpoint_path="final0",
        device="cpu",
        float16=False,
        postprocessor: PostprocessorInterface = None,
        dbn: bool = False  # Deprecated flag
    ):
        super().__init__(checkpoint_path, device, float16)

        if dbn:  # Handle deprecated flag for backward compatibility
            import warnings
            warnings.warn("The 'dbn' flag is deprecated...", DeprecationWarning)
            if postprocessor is None:
                # Lazy import of DbnPostprocessor only when needed
                from beat_this.model.madmom_postprocessor import DbnPostprocessor
                model_fps = self.spect.sample_rate / self.spect.hop_length
                postprocessor = DbnPostprocessor(fps=int(model_fps))

        if postprocessor is not None:
            self.frames2beats = postprocessor
        else:  # Default to MinimalPostprocessor
            model_fps = self.spect.sample_rate / self.spect.hop_length
            self.frames2beats = MinimalPostprocessor(fps=int(model_fps))

    def __call__(self, signal, sr):
        beat_logits, downbeat_logits = super().__call__(signal, sr)
        return self.frames2beats(beat_logits, downbeat_logits, padding_mask=None)
```

### Testing
1. Create test file `tests/test_inference.py`:
```python
# tests/test_inference.py
import pytest
import torch
import numpy as np
from beat_this.inference import Audio2Beats
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor

@pytest.fixture
def sample_audio():
    """Create sample audio data for testing."""
    return (
        np.random.randn(44100),  # 1 second of audio at 44.1kHz
        44100  # sample rate
    )

def test_audio2beats_default_postprocessor(sample_audio):
    """Test Audio2Beats with default (minimal) postprocessor."""
    model = Audio2Beats(device="cpu")
    signal, sr = sample_audio
    beat_times, downbeat_times = model(signal, sr)
    assert isinstance(beat_times, list)
    assert isinstance(downbeat_times, list)
    assert isinstance(model.frames2beats, MinimalPostprocessor)

def test_audio2beats_custom_postprocessor(sample_audio):
    """Test Audio2Beats with custom postprocessor."""
    custom_processor = MinimalPostprocessor(fps=100)
    model = Audio2Beats(device="cpu", postprocessor=custom_processor)
    signal, sr = sample_audio
    beat_times, downbeat_times = model(signal, sr)
    assert model.frames2beats is custom_processor

def test_audio2beats_dbn_flag(sample_audio):
    """Test Audio2Beats with deprecated dbn flag."""
    with pytest.warns(DeprecationWarning):
        model = Audio2Beats(device="cpu", dbn=True)
    signal, sr = sample_audio
    beat_times, downbeat_times = model(signal, sr)
    assert isinstance(model.frames2beats, DbnPostprocessor)

def test_audio2beats_dbn_flag_with_postprocessor(sample_audio):
    """Test Audio2Beats with both dbn flag and postprocessor."""
    custom_processor = MinimalPostprocessor(fps=100)
    with pytest.warns(DeprecationWarning):
        model = Audio2Beats(device="cpu", dbn=True, postprocessor=custom_processor)
    assert model.frames2beats is custom_processor  # Custom postprocessor should take precedence
```

## Step 4: Update the Command-Line Interface

### Goal
Allow users to select the postprocessor type and configure DBN parameters via command-line arguments.

### Files Affected
- `beat_this/cli.py`

### Changes
```python
# beat_this/cli.py
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor

def get_parser():
    parser.add_argument(
        "--postprocessor",
        type=str,
        choices=['minimal', 'dbn'],
        default='minimal',
        help="Which postprocessor to use (default: %(default)s)."
    )
    parser.add_argument(
        "--dbn-beats-per-bar",
        type=int,
        nargs='+',
        default=[3, 4],
        help="Possible beats per bar for DBN postprocessor (default: %(default)s)."
    )
    parser.add_argument(
        "--dbn-min-bpm",
        type=float,
        default=55.0,
        help="Minimum BPM for DBN postprocessor (default: %(default)s)."
    )
    parser.add_argument(
        "--dbn-max-bpm",
        type=float,
        default=215.0,
        help="Maximum BPM for DBN postprocessor (default: %(default)s)."
    )
    parser.add_argument(
        "--dbn-transition-lambda",
        type=float,
        default=100.0,
        help="Transition lambda for DBN postprocessor (default: %(default)s)."
    )
    return parser

def run(...):
    # Create postprocessor based on arguments
    model_fps = 50  # Placeholder - ideally get this dynamically

    if postprocessor == 'minimal':
        postproc_instance = MinimalPostprocessor(fps=model_fps)
    elif postprocessor == 'dbn':
        postproc_instance = DbnPostprocessor(
            fps=model_fps,
            beats_per_bar=dbn_beats_per_bar,
            min_bpm=dbn_min_bpm,
            max_bpm=dbn_max_bpm,
            transition_lambda=dbn_transition_lambda
        )
    else:
        raise ValueError(f"Unknown postprocessor type: {postprocessor}")

    file2file = File2File(model, device, float16, postprocessor=postproc_instance)
```

### Testing
1. Create test file `tests/test_cli.py`:
```python
# tests/test_cli.py
import pytest
from beat_this.cli import get_parser, run
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor

def test_cli_parser_defaults():
    """Test CLI parser default values."""
    parser = get_parser()
    args = parser.parse_args([])
    assert args.postprocessor == 'minimal'
    assert args.dbn_beats_per_bar == [3, 4]
    assert args.dbn_min_bpm == 55.0
    assert args.dbn_max_bpm == 215.0
    assert args.dbn_transition_lambda == 100.0

def test_cli_parser_custom_values():
    """Test CLI parser with custom values."""
    parser = get_parser()
    args = parser.parse_args([
        '--postprocessor', 'dbn',
        '--dbn-beats-per-bar', '4',
        '--dbn-min-bpm', '60.0',
        '--dbn-max-bpm', '180.0',
        '--dbn-transition-lambda', '50.0'
    ])
    assert args.postprocessor == 'dbn'
    assert args.dbn_beats_per_bar == [4]
    assert args.dbn_min_bpm == 60.0
    assert args.dbn_max_bpm == 180.0
    assert args.dbn_transition_lambda == 50.0

def test_cli_run_minimal_postprocessor(tmp_path):
    """Test CLI run with minimal postprocessor."""
    # Create temporary input file
    input_file = tmp_path / "test.wav"
    input_file.write_bytes(b"dummy wav data")
    
    args = get_parser().parse_args([
        str(input_file),
        '--postprocessor', 'minimal'
    ])
    # Mock the actual processing to avoid file I/O
    with pytest.MonkeyPatch.context() as m:
        m.setattr('beat_this.inference.File2File', lambda *args, **kwargs: None)
        run(args)

def test_cli_run_dbn_postprocessor(tmp_path):
    """Test CLI run with DBN postprocessor."""
    input_file = tmp_path / "test.wav"
    input_file.write_bytes(b"dummy wav data")
    
    args = get_parser().parse_args([
        str(input_file),
        '--postprocessor', 'dbn',
        '--dbn-beats-per-bar', '4'
    ])
    with pytest.MonkeyPatch.context() as m:
        m.setattr('beat_this.inference.File2File', lambda *args, **kwargs: None)
        run(args)
```

## Step 5: Update the PyTorch Lightning Module

### Goal
Allow the PLBeatThis module to use a configurable postprocessor for evaluation.

### Files Affected
- `beat_this/model/pl_module.py`

### Changes
```python
# beat_this/model/pl_module.py
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor

class PLBeatThis(LightningModule):
    def __init__(
        self,
        fps=50,
        use_dbn_eval=False,
        eval_dbn_beats_per_bar=(3, 4),
        eval_dbn_min_bpm=55.0,
        eval_dbn_max_bpm=215.0,
        eval_dbn_transition_lambda=100,
        eval_trim_beats=5,
        # ... other hparams
    ):
        super().__init__()
        self.save_hyperparameters()
        self.fps = fps

        # Configure evaluation postprocessor
        if use_dbn_eval:
            self.eval_postprocessor = DbnPostprocessor(
                fps=self.fps,
                beats_per_bar=eval_dbn_beats_per_bar,
                min_bpm=eval_dbn_min_bpm,
                max_bpm=eval_dbn_max_bpm,
                transition_lambda=eval_dbn_transition_lambda
            )
        else:
            self.eval_postprocessor = MinimalPostprocessor(fps=self.fps)

        self.eval_trim_beats = eval_trim_beats
        self.metrics = Metrics(eval_trim_beats=eval_trim_beats)
```

### Testing
1. Create test file `tests/test_pl_module.py`:
```python
# tests/test_pl_module.py
import pytest
import torch
from beat_this.model.pl_module import PLBeatThis
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor

@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    return {
        "audio": torch.randn(2, 44100),
        "beat": torch.randn(2, 100, 2),
        "downbeat": torch.randn(2, 100, 2),
        "padding_mask": torch.ones(2, 100, dtype=torch.bool)
    }

def test_pl_module_minimal_postprocessor(sample_batch):
    """Test PLBeatThis with minimal postprocessor."""
    model = PLBeatThis(fps=50, use_dbn_eval=False)
    assert isinstance(model.eval_postprocessor, MinimalPostprocessor)
    
    # Test validation step
    metrics = model.validation_step(sample_batch, 0)
    assert isinstance(metrics, dict)
    assert "val_loss" in metrics

def test_pl_module_dbn_postprocessor(sample_batch):
    """Test PLBeatThis with DBN postprocessor."""
    model = PLBeatThis(
        fps=50,
        use_dbn_eval=True,
        eval_dbn_beats_per_bar=(4,),
        eval_dbn_min_bpm=60.0,
        eval_dbn_max_bpm=180.0,
        eval_dbn_transition_lambda=50
    )
    assert isinstance(model.eval_postprocessor, DbnPostprocessor)
    assert model.eval_postprocessor.beats_per_bar == (4,)
    assert model.eval_postprocessor.min_bpm == 60.0
    assert model.eval_postprocessor.max_bpm == 180.0
    assert model.eval_postprocessor.transition_lambda == 50

    # Test validation step
    metrics = model.validation_step(sample_batch, 0)
    assert isinstance(metrics, dict)
    assert "val_loss" in metrics

def test_pl_module_predict_step(sample_batch):
    """Test PLBeatThis predict step with both postprocessors."""
    # Test with minimal postprocessor
    model_minimal = PLBeatThis(fps=50, use_dbn_eval=False)
    metrics_minimal, pred_minimal, dataset, spect_path = model_minimal.predict_step(sample_batch, 0)
    assert isinstance(metrics_minimal, dict)
    assert isinstance(pred_minimal, dict)
    
    # Test with DBN postprocessor
    model_dbn = PLBeatThis(fps=50, use_dbn_eval=True)
    metrics_dbn, pred_dbn, dataset, spect_path = model_dbn.predict_step(sample_batch, 0)
    assert isinstance(metrics_dbn, dict)
    assert isinstance(pred_dbn, dict)
```

## Step 6: Update Launch Scripts

### Goal
Configure the postprocessor settings when instantiating PLBeatThis in the training and evaluation scripts.

### Files Affected
- `beat_this/launch_scripts/train.py`
- `beat_this/launch_scripts/compute_paper_metrics.py`

### Changes
```python
# beat_this/launch_scripts/train.py
def main(args):
    # Instantiate PLBeatThis with evaluation postprocessor parameters
    pl_model = PLBeatThis(
        fps=args.fps,
        use_dbn_eval=args.dbn,
        # Add arguments for DBN evaluation parameters if needed
    )

# beat_this/launch_scripts/compute_paper_metrics.py
def plmodel_setup(checkpoint, eval_trim_beats, dbn, gpu):
    hparams = checkpoint["hyper_parameters"]
    if eval_trim_beats is not None:
        hparams["eval_trim_beats"] = eval_trim_beats
    if dbn is not None:
        hparams["use_dbn_eval"] = dbn

    model = PLBeatThis(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    return model, trainer
```

### Testing
1. Create test file `tests/test_launch_scripts.py`:
```python
# tests/test_launch_scripts.py
import pytest
from beat_this.launch_scripts.train import main as train_main
from beat_this.launch_scripts.compute_paper_metrics import plmodel_setup

def test_train_script_minimal_postprocessor(tmp_path):
    """Test training script with minimal postprocessor."""
    # Create temporary checkpoint
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"dummy checkpoint data")
    
    args = type('Args', (), {
        'checkpoint': str(checkpoint),
        'dbn': False,
        'fps': 50
    })
    
    with pytest.MonkeyPatch.context() as m:
        # Mock PyTorch Lightning components
        m.setattr('pytorch_lightning.Trainer.fit', lambda *args, **kwargs: None)
        m.setattr('pytorch_lightning.Trainer.test', lambda *args, **kwargs: None)
        train_main(args)

def test_train_script_dbn_postprocessor(tmp_path):
    """Test training script with DBN postprocessor."""
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"dummy checkpoint data")
    
    args = type('Args', (), {
        'checkpoint': str(checkpoint),
        'dbn': True,
        'fps': 50
    })
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr('pytorch_lightning.Trainer.fit', lambda *args, **kwargs: None)
        m.setattr('pytorch_lightning.Trainer.test', lambda *args, **kwargs: None)
        train_main(args)

def test_compute_metrics_setup():
    """Test compute_paper_metrics setup function."""
    # Mock checkpoint data
    checkpoint = {
        'hyper_parameters': {
            'fps': 50,
            'use_dbn_eval': False
        },
        'state_dict': {}
    }
    
    model, trainer = plmodel_setup(checkpoint, eval_trim_beats=5, dbn=False, gpu=False)
    assert not model.hparams.use_dbn_eval
    assert model.eval_trim_beats == 5

    # Test with DBN enabled
    checkpoint['hyper_parameters']['use_dbn_eval'] = True
    model, trainer = plmodel_setup(checkpoint, eval_trim_beats=5, dbn=True, gpu=False)
    assert model.hparams.use_dbn_eval
```

## Testing and Validation

After implementing each step, thorough testing should be performed to ensure:
1. Core functionality remains intact
2. All postprocessor implementations work correctly
3. Command-line interface changes work as expected
4. Training and evaluation scripts function properly with the new postprocessor configuration
5. Backward compatibility is maintained where needed

### Running Tests
1. Install test dependencies:
```bash
pip install pytest pytest-mock pytest-cov
```

2. Run all tests:
```bash
pytest tests/ -v --cov=beat_this
```

3. Generate coverage report:
```bash
pytest tests/ --cov=beat_this --cov-report=html
```

### Test Coverage Goals
- Interface definition: 100%
- Postprocessor implementations: >90%
- Inference classes: >90%
- CLI: >85%
- PyTorch Lightning module: >85%
- Launch scripts: >80%

## Notes
- The refactoring maintains backward compatibility through the deprecated `dbn` flag
- All new parameters have sensible defaults
- The interface design allows for easy addition of new postprocessor implementations in the future
- The madmom-based postprocessor is now in a separate file, making it easier to:
  - Manage dependencies (madmom is only required when using DbnPostprocessor)
  - Add alternative postprocessor implementations
  - Test each postprocessor independently
  - Maintain and update the madmom integration separately