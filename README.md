## Notes on the Grading Test

### Basics

**Training**

```plain
python --data_path=<...> [--eval_path=<...>]
```

**Inference**

```plain
python --data_path=<...> [--model_path=<...>]
```

If model is not specified, it defaults to `model/model.pickle`.

### Pretrained models

- `model/model.pickle` is trained on the full set without real-time evaluation. The best model (evaluated on the full set) across 5 independent runs is saved.
- `model/model-senza-<circuit>.pickle` is trained on the set with `<circuit>` excluded (also without real-time evaluation). The best model (evaluated on `<circuit>`) across 5 independent runs is saved.

The score of the 5 runs can be found in the report. The script for conducting the 5 runs can be found in `stat/grade.py`.

### Precautions

- Despite efforts to stablize training, the training is still somehow unstable. It is recommended to train and evaluate it for multiple times and calculate the average score.
- If you see `[ ! ] The training loss is NaN. Numerical collapse has occured.` during training, some rare numerical explosion must have happened. In this case, the saved model will be corrupt and NaNNy. DO NOT use it but train a new one instead.
- Due to lack of edge case detections, the training might fail if the training set is too small.

