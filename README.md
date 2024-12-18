# Graph Neural Networks for Mid-Price Movement Forecasting in Limit Order Books

### Authors: Božo Đerek, Benedicte Gabelica, Karlo Vrančić

---

## Introduction
Time-series forecasting using Limit Order Books (LOBs) plays a crucial role when forecasting short-term stock price movements. Specifically, predicted metrics such as mid-price movement or volatility contain valuable information for traders and investors and can be of great help when analyzing the market. Therefore, it is important to identify the most effective machine learning models and data representation for accurate forecasting.

In this paper, we propose representing LOBs as graphs and use Graph Neural Networks (GNNs) to predict mid-price movement. The models we used to forecast mid-price movement are adaptations of [MTGNN](https://arxiv.org/abs/2005.11650) and [StemGNN](https://arxiv.org/abs/2103.07719). The code used for all calculations is given in this repository.

---

## Instructions to Run Code

### MTGNN


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

