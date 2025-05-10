import pytest
from beat_this.cli import get_parser, run
from beat_this.model.postprocessor import MinimalPostprocessor
from beat_this.model.madmom_postprocessor import DbnPostprocessor
import warnings


def test_cli_parser_defaults():
    """Test CLI parser default values."""
    parser = get_parser()
    # Add dummy input file to satisfy required argument
    args = parser.parse_args(['dummy.wav'])
    assert args.postprocessor == 'minimal'
    assert args.dbn_beats_per_bar == [3, 4]
    assert args.dbn_min_bpm == 55.0
    assert args.dbn_max_bpm == 215.0
    assert args.dbn_transition_lambda == 100.0


def test_cli_parser_custom_values():
    """Test CLI parser with custom values."""
    parser = get_parser()
    args = parser.parse_args([
        'dummy.wav',  # Required input argument
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


def test_cli_deprecated_dbn_flag():
    """Test CLI parser with deprecated dbn flag."""
    parser = get_parser()
    # Just verify the flag is set correctly without expecting a warning at parsing time
    args = parser.parse_args(['dummy.wav', '--dbn'])
    assert args.dbn is True
    
    # Test dbn flag precedence
    args = parser.parse_args(['dummy.wav', '--dbn', '--postprocessor', 'minimal'])
    assert args.dbn is True
    assert args.postprocessor == 'minimal'


def test_cli_run_minimal_postprocessor(tmp_path, monkeypatch):
    """Test CLI run with minimal postprocessor."""
    # Create temporary input file
    input_file = tmp_path / "test.wav"
    input_file.write_bytes(b"dummy wav data")
    
    # Create a parser and add the test input file
    parser = get_parser()
    args = parser.parse_args([
        str(input_file),
        '--postprocessor', 'minimal'
    ])
    
    # Mock file2file to avoid actual processing
    class MockFile2File:
        def __init__(self, model, device, float16, postprocessor=None):
            self.model = model
            self.device = device
            self.float16 = float16
            self.postprocessor = postprocessor
            
        def __call__(self, input_path, output_path):
            # Just verify the postprocessor type
            assert isinstance(self.postprocessor, MinimalPostprocessor)
    
    monkeypatch.setattr('beat_this.cli.File2File', MockFile2File)
    
    # Run with the mocked File2File
    run(**vars(args))


def test_cli_run_dbn_postprocessor(tmp_path, monkeypatch):
    """Test CLI run with DBN postprocessor."""
    # Create temporary input file
    input_file = tmp_path / "test.wav"
    input_file.write_bytes(b"dummy wav data")
    
    # Create a parser and add the test input file
    parser = get_parser()
    args = parser.parse_args([
        str(input_file),
        '--postprocessor', 'dbn',
        '--dbn-beats-per-bar', '4'
    ])
    
    # Mock file2file to avoid actual processing
    class MockFile2File:
        def __init__(self, model, device, float16, postprocessor=None):
            self.model = model
            self.device = device
            self.float16 = float16
            self.postprocessor = postprocessor
            
        def __call__(self, input_path, output_path):
            # Verify the postprocessor type and parameters
            assert isinstance(self.postprocessor, DbnPostprocessor)
            assert self.postprocessor.beats_per_bar == (4,)
    
    monkeypatch.setattr('beat_this.cli.File2File', MockFile2File)
    
    # Run with the mocked File2File
    run(**vars(args))


def test_cli_run_deprecated_dbn_flag(tmp_path, monkeypatch):
    """Test CLI run with deprecated dbn flag."""
    # Create temporary input file
    input_file = tmp_path / "test.wav"
    input_file.write_bytes(b"dummy wav data")
    
    # Create a parser and add the test input file with the deprecated flag
    parser = get_parser()
    args = parser.parse_args([
        str(input_file),
        '--dbn'
    ])
    
    # Mock file2file to avoid actual processing
    class MockFile2File:
        def __init__(self, model, device, float16, postprocessor=None):
            self.model = model
            self.device = device
            self.float16 = float16
            self.postprocessor = postprocessor
            
        def __call__(self, input_path, output_path):
            # Verify that we got a DbnPostprocessor despite using the deprecated flag
            assert isinstance(self.postprocessor, DbnPostprocessor)
    
    monkeypatch.setattr('beat_this.cli.File2File', MockFile2File)
    
    # Run with the mocked File2File - should give a deprecation warning
    with pytest.warns(DeprecationWarning):
        run(**vars(args))


def test_cli_run_multiple_files(tmp_path, monkeypatch):
    """Test CLI run with multiple input files and directory."""
    # Create temporary directory structure with test files
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    input_file1 = tmp_path / "test1.wav"
    input_file2 = test_dir / "test2.wav"
    input_file1.write_bytes(b"dummy wav data 1")
    input_file2.write_bytes(b"dummy wav data 2")
    
    # Create a parser and add multiple test files
    parser = get_parser()
    args = parser.parse_args([
        str(input_file1),
        str(test_dir),
        '--postprocessor', 'dbn',
        '--dbn-beats-per-bar', '3', '4',
        '--dbn-min-bpm', '60.0',
        '--dbn-max-bpm', '180.0'
    ])
    
    # Mock file2file to avoid actual processing
    processed_files = []
    class MockFile2File:
        def __init__(self, model, device, float16, postprocessor=None):
            self.model = model
            self.device = device
            self.float16 = float16
            self.postprocessor = postprocessor
            # Verify the postprocessor parameters
            assert isinstance(self.postprocessor, DbnPostprocessor)
            assert self.postprocessor.beats_per_bar == (3, 4)
            assert self.postprocessor.min_bpm == 60.0
            assert self.postprocessor.max_bpm == 180.0
            
        def __call__(self, input_path, output_path):
            processed_files.append(str(input_path))
    
    # Properly mock tqdm
    class MockTqdm:
        @staticmethod
        def tqdm(iterable):
            return iterable
    
    monkeypatch.setattr('beat_this.cli.File2File', MockFile2File)
    monkeypatch.setattr('beat_this.cli.tqdm', MockTqdm)
    
    # Run with the mocked File2File
    run(**vars(args))
    
    # Verify that both files were processed
    assert len(processed_files) == 2
    assert str(input_file1) in processed_files
    assert str(input_file2) in processed_files


def test_postprocessor_exception(tmp_path, monkeypatch):
    """Test CLI handling of invalid postprocessor type."""
    # Create temporary input file
    input_file = tmp_path / "test.wav"
    input_file.write_bytes(b"dummy wav data")
    
    # Create a parser with invalid postprocessor (shouldn't happen with choices constrained)
    parser = get_parser()
    args = parser.parse_args([str(input_file)])
    # Manually set an invalid postprocessor value
    args_dict = vars(args)
    args_dict['postprocessor'] = 'invalid_type'
    
    # Mock file2file to avoid actual processing
    monkeypatch.setattr('beat_this.cli.File2File', lambda *args, **kwargs: None)
    
    # Run with the invalid postprocessor type
    with pytest.raises(ValueError, match="Unknown postprocessor type: invalid_type"):
        run(**args_dict) 