# BPM Finder — User Guide

Extract the complete Windows ZIP, then open **BPM Finder.exe**. Keep `_internal` and `models` beside it. No Python installation or command prompt is required.

## Language

English is the default. Use **Language** at the top of the window to select **עברית** (Hebrew). The choice is remembered. Changing languages restarts the window, preserves the input queue and analysis settings, and applies the correct reading direction. Switching is disabled while analysis is running.

## Analyze music

1. Select **Add files** or **Add folder**, or drag files and folders into the window.
2. Choose a detection preset and output location. **Standard — high accuracy** and **Fast — small model** are included.
3. Select **Start analysis**. Each row shows its status, estimated BPM, beats and downbeats.
4. Use **Open output folder**, or double-click a completed result to see its tempo chart.

The `.beats` output contains beat times in seconds and beat positions within the bar. BPM is estimated from the median interval between detected beats. The source audio is not changed.

**Advanced settings** exposes the original suffix, append, skip-existing, reserve-file, DBN, compute-device, Float16 and raw NPY options. The standalone Windows package uses its own CPU engine and includes DBN. GPU operation requires a compatible source environment.

## History and cache

**Remember song analyses in cache** is enabled by default. Identical file contents, model weights and analysis settings reuse saved results, even after a rename or after previous output files have been removed. Changed audio, model weights or inference settings require fresh analysis. Requesting raw NPY data requires a saved raw result or a new analysis.

**History and cache…** offers search, saved tempo charts, beat export, adding source songs to the queue, and deleting cached results. Clearing the cache leaves audio and exported files in place. **Analyze again** bypasses both the cache and Skip existing output. Cache and language preferences persist in `%LOCALAPPDATA%\BPM Finder`. Existing results created before caching was added are not imported automatically.

## Tempo changes

Use **BPM chart and changes** to view tempo throughout a song. Adjust sensitivity or advanced thresholds, hover for time/BPM, select an event to focus, use the wheel or plus/minus to zoom, and double-click for the full song. Export tempo changes or the curve as CSV, or the current chart as PNG.

Abrupt changes, gradual accelerations and slowdowns are estimates from detected beats. Detection errors and double/half tempo can affect results. Changing chart sensitivity does not rerun the model.

## Models and troubleshooting

**Models and downloads…** includes official variants, research models, compatible local checkpoints and direct model links. Downloads are verified before replacing an installed model. Audio analysis stays local.

Use **Diagnostics and repair** to check the engine, DBN, audio decoder and selected model. Repair a missing model with **Download again / Repair**. For missing program files, extract the full Windows package again. **Run details** contains per-file errors; other files continue processing.

For source development, extract the source ZIP separately and run **Setup UI.cmd**. It prepares an isolated Python 3.12 CPU environment and creates a launcher.
