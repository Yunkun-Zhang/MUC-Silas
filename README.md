# MUC-Silas
Implementation of paper [*MUC-driven Feature Importance Measurement and Adversarial Analysis for Random Forest*](https://arxiv.org/abs/2202.12512v1) based on [Silas](https://www.depintel.com/) models. Cite the paper with

```
@misc{ma2022mucdriven,
      title={MUC-driven Feature Importance Measurement and Adversarial Analysis for Random Forest}, 
      author={Shucen Ma and Jianqi Shi and Yanhong Huang and Shengchao Qin and Zhe Hou},
      year={2022},
      eprint={2202.12512},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Running

1. Clone this repository and `cd` into it.

2. Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3. Run `main.py` with your Silas model path and test data. Add `-S` for M-Shapley values, and `-A` for adversarial samples:
    ```shell
   python main.py -m model_path -t test_file [-S] [-A] 
   ```

NOTE: Please ensure that the order of features in the testing data file matches the order of features in the metadata file.

### Outputs

For M-Shapley value, the result is the computed values for each class for each feature.

For adversarial sample, the result is the opt adv sample only for the first data instance of test file to save time. You can modify this in `main.py`.

Results will be printed.
