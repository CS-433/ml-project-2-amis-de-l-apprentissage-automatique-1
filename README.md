# Graph Neural Networks for Mid-Price Movement Forecasting in Limit Order Books

### Authors: Božo Đerek, Benedicte Gabelica, Karlo Vrančić

---

## Introduction
Predicting mid-price movements is a critical challenge in financial markets, as accurate predictions are important for good trading strategies and market efficiency. The mid-price, derived from the best bid and ask prices, is an essential indicator of market trends. High-frequency limit order book data provides a rich source of information for predicting these movements, but the data's complexity and imbalance pose significant challenges.

This project addresses the problem of mid-price movement prediction as a ternary classification task, where the goal is to determine whether the mid-price will move up, down, or remain stationary over a given time horizon. We apply Graph Neural Networks (GNNs), which are well-suited for modeling complex relational data. We train the model on a commonly used benchmark LOB dataset and evaluate our results against established models in the literature. Additionally, we tackle the challenge of label imbalance, a frequent issue in financial datasets, to improve prediction accuracy.

The models we use to forecast mid-price movement are adaptations of [MTGNN](https://arxiv.org/abs/2005.11650) and [StemGNN](https://arxiv.org/abs/2103.07719). The code used for all calculations is given in this repository.

---

## Instructions to Run Code
**Note**: The `.csv` files found in `train_test.zip` were created using the `preprocess-fi-2010.ipynb` notebook ([link](https://www.kaggle.com/code/bderek81/preprocess-fi-2010)). They are a subset of the columns from the [FI-2010](http://dx.doi.org/10.1002/for.2543) dataset (only prices and volumes for 10 bid and ask levels were kept as features - 40 in total, as well as the labels for all 5 horizons). The exact version of the original dataset we use can be found [here](https://www.kaggle.com/datasets/bderek81/fi2010), while the preprocessed version can be found [here](https://www.kaggle.com/datasets/bderek81/fi2010-v2).

### MTGNN
We recommend running `mtgnn-main.ipynb` on Kaggle (with GPU T4 x2), as that is the exact setup we used. You can find the notebook [here](https://www.kaggle.com/code/bderek81/mtgnn-main). If you really wish to run locally, the imports are all on top of the notebook (pretty standard Python packages). Besides that, the only thing you need to take care of is to change the path where the code looks for `train.csv` and `test.csv` and make sure the files are present there.


---

### StemGNN
We adapted the StemGNN to predict mid-price movement. To start training the model, please follow these steps:

1. **Navigate to the StemGNN directory**:
   ```bash
   cd StemGNN
   ```

2. **Install necessary requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the train and test datasets with labels for the desired horizon**:
   Execute the following command:
   ```bash
   python build_dataset.py --horizon 10
   ```
   Replace `10` with the desired horizon value (1, 2, 3, 5, or 10).

4. **Start training the model**:
   Run the following command:
   ```bash
   python main.py --train True --targets targets --evaluate True --dataset train_test_modified --epoch 15 --device cuda --window_size 10 --horizon 10
   ```
   - Adjust the following arguments as needed:
     - `--epoch`: Number of training epochs (default: 20).
     - `--device`: Use `cuda` for GPU or `cpu` if no GPU is available.
     - `--window_size`: Number of rows used to predict one label (default: 10).
     - `--horizon`: Use the same horizon value specified in step 3.

---

