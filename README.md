# Beat This
Accurate and general beat tracker from the paper ...

## Compute beats for a audio file

 
```
python launch_scripts/predict.py --model final0 --audio-path path_to_audio_file
```

Set ```--gpu -1``` if you want to run on cpu.
You can add ```--dbn``` if you want to use the DBN. The DBN parameters are the default one from madmom.

Please be aware that the model ```final0``` was trained on all data except the GTZAN dataset. So if you run the inference on some data that was used for training, the results may be unfairly good.

## Experiment reproducibility
Compute results on the test set GTZAN

```
python launch_scripts/compute_paper_metrics.py --models final0 final1 final2 --datasplit test
```

Compute k-fold
```
python launch_scripts/compute_paper_metrics.py --aggregation-type k-fold --models checkpoints/fold0.ckpt checkpoints/fold1.ckpt checkpoints/fold2.ckpt checkpoints/fold3.ckpt checkpoints/fold4.ckpt checkpoints/fold5.ckpt checkpoints/fold6.ckpt checkpoints/fold7.ckpt 
```