Environment setup:

- Python 3.10.5, with numpy installed.
- See `requirements.txt` for the exact list of packages in the environment.
- Currently the program does not use GPU. No GPU is needed to run it.

Experiment setup:

- My training command: `python main.py --data_path=datasets/senza-hyp --eval_path=datasets/hyp`, where `datasets/senza-hyp` contains every circuit except `hyp` while `datasets/hyp` contains only `hyp`.
- The training process does not terminate by itself. Modify the file `trace/termination_flag` and save it to terminate the training process and save the model to `model/model.pickle`.

Current results (outdated):

- With the training setup, the model can eventually achieve 93% correctness and ~0.32 f1-score on the evaluation samples.

What to do next:

- Improve stability and usability
  - ~~Automatically detect that the training has converged and stop it at an appropriate point without human intervention.~~
  - **[★]** Test the training process on other train & eval splits.
- Improve performence
  - Implement more optimization approaches, e.g. ~~dropout~~, ~~batch normalization~~, residual connections, **[★]** better initialization approaches, **[★]** gradient descent with momentum, ~~better loss function~~, ~~better learning rate scheduling~~. To avoid waiting time, please assess their effectiveness using your knowledge before using them. If you have other ideas, you may try them as well.
  - **[★]** Get to know what the features actually mean in the dataset and implement some problem-specific pre-processing techniques.
  - Audit the code and fix potential bugs in calculations.
  - Adjust hyperparameters.
  - Adjust network structure.

Important notice:

- During the actual test, the user will only specify `data_path` and not the `eval_path` during training. The training algorithm must work without a separate evaluation set.
- It is important to also retain the stability of training.
