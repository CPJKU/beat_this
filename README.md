# Beat This
Accurate and general beat tracker from the paper ...

## Compute beats for a audio file

 ```python launch_scripts/predict.py --model https://cloud.cp.jku.at/index.php/s/Dbtd47JqzDxWoks/download/final0.ckpt --gpu 0 --audio-path path_to_audio_file
```

You can add ```--dbn``` if you want to use the DBN. The DBN parameters are the default one from madmom.

## Experiment reproducibility
Compute results on the test set GTZAN

```
python launch_scripts/compute_paper_metrics.py --models https://cloud.cp.jku.at/index.php/s/Dbtd47JqzDxWoks/download/final0.ckpt https://cloud.cp.jku.at/index.php/s/DCm9YLkTBAEc4y3/download/final1.ckpt https://cloud.cp.jku.at/index.php/s/E8A3McdxpwSGGwJ/download/final2.ckpt --gpu 0 --datasplit test
```
