# tests/test_launch_scripts.py
import pytest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import launch_scripts directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from launch_scripts.train import main as train_main
from launch_scripts.compute_paper_metrics import plmodel_setup

def test_train_script_minimal_postprocessor(tmp_path):
    """Test training script with minimal postprocessor."""
    # Create temporary checkpoint
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"dummy checkpoint data")
    
    args = type('Args', (), {
        'checkpoint': str(checkpoint),
        'dbn': False,
        'use_dbn_eval': False,
        'eval_dbn_beats_per_bar': (3, 4),
        'eval_dbn_min_bpm': 55.0,
        'eval_dbn_max_bpm': 215.0,
        'eval_dbn_transition_lambda': 100.0,
        'fps': 50,
        'seed': 0,
        'name': 'test',
        'force_flash_attention': False,
        'compile': [],
        'n_layers': 6,
        'transformer_dim': 512,
        'frontend_dropout': 0.1,
        'transformer_dropout': 0.2,
        'lr': 0.0008,
        'weight_decay': 0.01,
        'logger': 'none',
        'num_workers': 8,
        'n_heads': 16,
        'loss': 'shift_tolerant_weighted_bce',
        'warmup_steps': 1000,
        'max_epochs': 100,
        'batch_size': 8,
        'accumulate_grad_batches': 8,
        'train_length': 1500,
        'eval_trim_beats': 5,
        'val_frequency': 5,
        'tempo_augmentation': True,
        'pitch_augmentation': True,
        'mask_augmentation': True,
        'sum_head': True,
        'partial_transformers': True,
        'length_based_oversampling_factor': 0.65,
        'val': True,
        'fold': None,
        'hung_data': False,
        'gpu': 0
    })
    
    with pytest.MonkeyPatch.context() as m:
        # Mock PyTorch Lightning components
        m.setattr('pytorch_lightning.Trainer.fit', lambda *args, **kwargs: None)
        m.setattr('pytorch_lightning.Trainer.test', lambda *args, **kwargs: None)
        # Mock datamodule to avoid file I/O
        m.setattr('beat_this.dataset.BeatDataModule.setup', lambda *args, **kwargs: None)
        m.setattr('beat_this.dataset.BeatDataModule.get_train_positive_weights',
                  lambda *args, **kwargs: {"beat": 1.0, "downbeat": 1.0})
        
        # Import here to avoid module loading issues with the monkeypatching
        import torch
        train_main(args)

def test_train_script_dbn_postprocessor(tmp_path):
    """Test training script with DBN postprocessor."""
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"dummy checkpoint data")
    
    args = type('Args', (), {
        'checkpoint': str(checkpoint),
        'dbn': False,  # Should be ignored since use_dbn_eval is set
        'use_dbn_eval': True,
        'eval_dbn_beats_per_bar': (4,),
        'eval_dbn_min_bpm': 60.0,
        'eval_dbn_max_bpm': 180.0,
        'eval_dbn_transition_lambda': 50.0,
        'fps': 50,
        'seed': 0,
        'name': 'test',
        'force_flash_attention': False,
        'compile': [],
        'n_layers': 6,
        'transformer_dim': 512,
        'frontend_dropout': 0.1,
        'transformer_dropout': 0.2,
        'lr': 0.0008,
        'weight_decay': 0.01,
        'logger': 'none',
        'num_workers': 8,
        'n_heads': 16,
        'loss': 'shift_tolerant_weighted_bce',
        'warmup_steps': 1000,
        'max_epochs': 100,
        'batch_size': 8,
        'accumulate_grad_batches': 8,
        'train_length': 1500,
        'eval_trim_beats': 5,
        'val_frequency': 5,
        'tempo_augmentation': True,
        'pitch_augmentation': True,
        'mask_augmentation': True,
        'sum_head': True,
        'partial_transformers': True,
        'length_based_oversampling_factor': 0.65,
        'val': True,
        'fold': None,
        'hung_data': False,
        'gpu': 0
    })
    
    with pytest.MonkeyPatch.context() as m:
        # Mock PyTorch Lightning components
        m.setattr('pytorch_lightning.Trainer.fit', lambda *args, **kwargs: None)
        m.setattr('pytorch_lightning.Trainer.test', lambda *args, **kwargs: None)
        # Mock datamodule to avoid file I/O
        m.setattr('beat_this.dataset.BeatDataModule.setup', lambda *args, **kwargs: None)
        m.setattr('beat_this.dataset.BeatDataModule.get_train_positive_weights',
                  lambda *args, **kwargs: {"beat": 1.0, "downbeat": 1.0})
        
        # Import here to avoid module loading issues with the monkeypatching
        import torch
        train_main(args)

def test_train_script_deprecated_dbn_param(tmp_path):
    """Test training script with deprecated dbn parameter."""
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"dummy checkpoint data")
    
    args = type('Args', (), {
        'checkpoint': str(checkpoint),
        'dbn': True,
        'use_dbn_eval': None,  # Should be set to True from dbn
        'eval_dbn_beats_per_bar': (3, 4),
        'eval_dbn_min_bpm': 55.0,
        'eval_dbn_max_bpm': 215.0,
        'eval_dbn_transition_lambda': 100.0,
        'fps': 50,
        'seed': 0,
        'name': 'test',
        'force_flash_attention': False,
        'compile': [],
        'n_layers': 6,
        'transformer_dim': 512,
        'frontend_dropout': 0.1,
        'transformer_dropout': 0.2,
        'lr': 0.0008,
        'weight_decay': 0.01,
        'logger': 'none',
        'num_workers': 8,
        'n_heads': 16,
        'loss': 'shift_tolerant_weighted_bce',
        'warmup_steps': 1000,
        'max_epochs': 100,
        'batch_size': 8,
        'accumulate_grad_batches': 8,
        'train_length': 1500,
        'eval_trim_beats': 5,
        'val_frequency': 5,
        'tempo_augmentation': True,
        'pitch_augmentation': True,
        'mask_augmentation': True,
        'sum_head': True,
        'partial_transformers': True,
        'length_based_oversampling_factor': 0.65,
        'val': True,
        'fold': None,
        'hung_data': False,
        'gpu': 0
    })
    
    with pytest.MonkeyPatch.context() as m:
        # Mock PyTorch Lightning components
        m.setattr('pytorch_lightning.Trainer.fit', lambda *args, **kwargs: None)
        m.setattr('pytorch_lightning.Trainer.test', lambda *args, **kwargs: None)
        # Mock datamodule to avoid file I/O
        m.setattr('beat_this.dataset.BeatDataModule.setup', lambda *args, **kwargs: None)
        m.setattr('beat_this.dataset.BeatDataModule.get_train_positive_weights',
                  lambda *args, **kwargs: {"beat": 1.0, "downbeat": 1.0})
        
        # Import here to avoid module loading issues with the monkeypatching
        import torch
        # Test that a warning is raised
        with pytest.warns(DeprecationWarning):
            train_main(args)

def test_compute_metrics_setup():
    """Test compute_paper_metrics setup function."""
    # Mock checkpoint data
    checkpoint = {
        'hyper_parameters': {
            'fps': 50,
            'use_dbn_eval': False,
            'eval_dbn_beats_per_bar': (3, 4),
            'eval_dbn_min_bpm': 55.0,
            'eval_dbn_max_bpm': 215.0,
            'eval_dbn_transition_lambda': 100.0,
            'eval_trim_beats': 5,
            # Add other necessary hparams that PLBeatThis expects from the checkpoint
            'spect_dim': 128,
            'transformer_dim': 512,
            'ff_mult': 4,
            'n_layers': 6,
            'stem_dim': 32,
            'dropout': {'frontend': 0.1, 'transformer': 0.2},
            'lr': 0.0008,
            'weight_decay': 0.01,
            'pos_weights': {'beat': 1, 'downbeat': 1}, # Or a tensor if that's what your setup expects
            'head_dim': 32,
            'loss_type': 'shift_tolerant_weighted_bce',
            'warmup_steps': 1000,
            'max_epochs': 100,
            'sum_head': True,
            'partial_transformers': True,
        },
        'state_dict': {}
    }
    
    # Mock PLBeatThis to avoid actual model instantiation and state_dict loading issues
    with pytest.MonkeyPatch.context() as m:
        # Mock the PLBeatThis where plmodel_setup will look for it
        m.setattr('launch_scripts.compute_paper_metrics.PLBeatThis',
                  lambda **kwargs: type('MockModel', (),
                                       {'hparams': kwargs,
                                        'eval_trim_beats': kwargs.get('eval_trim_beats', 5), # Ensure default if not in kwargs
                                        'load_state_dict': lambda *args, **kwargs: None, # Changed to generic signature
                                        'cuda': lambda slf: slf # mock cuda call
                                        }
                                      )
                 )
        m.setattr('pytorch_lightning.Trainer', lambda **kwargs: None)
        
        # Test with minimal postprocessor (default)
        model, _ = plmodel_setup(checkpoint, eval_trim_beats=None, dbn=None, 
                               use_dbn_eval=None, eval_dbn_beats_per_bar=None,
                               eval_dbn_min_bpm=None, eval_dbn_max_bpm=None,
                               eval_dbn_transition_lambda=None, gpu=0)
        assert not model.hparams['use_dbn_eval']
        
        # Test with custom trimming
        model, _ = plmodel_setup(checkpoint, eval_trim_beats=10, dbn=None, 
                               use_dbn_eval=None, eval_dbn_beats_per_bar=None,
                               eval_dbn_min_bpm=None, eval_dbn_max_bpm=None,
                               eval_dbn_transition_lambda=None, gpu=0)
        assert model.eval_trim_beats == 10

        # Test with deprecated dbn parameter
        with pytest.warns(DeprecationWarning):
            model, _ = plmodel_setup(checkpoint, eval_trim_beats=None, dbn=True, 
                                   use_dbn_eval=None, eval_dbn_beats_per_bar=None,
                                   eval_dbn_min_bpm=None, eval_dbn_max_bpm=None,
                                   eval_dbn_transition_lambda=None, gpu=0)
            assert model.hparams['use_dbn_eval']

        # Test with both dbn and use_dbn_eval (use_dbn_eval should take precedence)
        with pytest.warns(DeprecationWarning):
            model, _ = plmodel_setup(checkpoint, eval_trim_beats=None, dbn=True, 
                                   use_dbn_eval=False, eval_dbn_beats_per_bar=None,
                                   eval_dbn_min_bpm=None, eval_dbn_max_bpm=None,
                                   eval_dbn_transition_lambda=None, gpu=0)
            assert not model.hparams['use_dbn_eval']

        # Test with custom DBN parameters
        model, _ = plmodel_setup(checkpoint, eval_trim_beats=None, dbn=None, 
                               use_dbn_eval=True, eval_dbn_beats_per_bar=(4,),
                               eval_dbn_min_bpm=60.0, eval_dbn_max_bpm=180.0,
                               eval_dbn_transition_lambda=50.0, gpu=0)
        assert model.hparams['use_dbn_eval']
        assert model.hparams['eval_dbn_beats_per_bar'] == (4,)
        assert model.hparams['eval_dbn_min_bpm'] == 60.0
        assert model.hparams['eval_dbn_max_bpm'] == 180.0
        assert model.hparams['eval_dbn_transition_lambda'] == 50.0 