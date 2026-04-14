# Changelog

All notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Clarified installation instructions for madmom and mir_eval
- Load checkpoints with `weights_only=True` when supported
- Fix checkpoint downloads after server-side update
- Provide separate `infer_beat_numbers()` function
- Command-line tool: Support saving raw activations / logits
- Training script: Support resuming from previous checkpoint
- Migrate to pyproject.toml (thanks to @JacobLinCool)
- Support non-CUDA accelerator chips (thanks to @tillt)

## [1.0] - 2024-10-18

- Initial release
