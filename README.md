# PSO-LSTM-for-Prediction
Prediction of the Stock Adjusted Closing Price Based on Improved PSO-LSTM Neural Network(Keras,Python)

The introduction of its principle can be seen in the paper: ''Prediction of the Stock Adjusted Closing Price Based on Improved PSO-LSTM Neural Network''

DOI : https://ieeexplore.ieee.org/document/9941330

![image](https://s2.loli.net/2024/10/12/NYG3wAmJzfkd65i.png)


The code model is based on LSTM neural network optimized by improved particle swarm optimization algorithm, which predicts the next day by Adj Close Price of the first 20 days.

Multi-dimensional and multi-step prediction can be achieved, seq_len in the code represents the length of the input sequence, mulpre represents the number of prediction steps. 

Code coverage detailed comments, can be modified as needed. 

Note that file_path and csv_path are the addresses of the folder where the results are saved, and filename is the address where the historical data set is stored. Please change it as required. l is the file name of each data set, using the for loop can achieve a run to predict multiple data sets, and save the results. 
