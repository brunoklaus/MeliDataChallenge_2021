# MeliDataChallenge_2021
My Solution for the MercadoLibre 2021 Data Challenge. Currently requires tqdm 4.6, lightgbm 3.1, tensorboard 2.6, Pytorch 1.8.1 + PyTorch Geometric 1.7.0 w/torch-sparse,torch-scatter (https://github.com/pyg-team/pytorch_geometric), as the custom DataLoader/Dataset interfaces helped me work faster. It can be a hassle to install, so I'll see if I can find a way around this. You may need a GPU with 10 GB VRAM, and also 16 GB RAM to run this without any problems. 
# Video of me going over the code for 30 minutes
https://youtu.be/TZbwB7Oh-zw
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/TZbwB7Oh-zw/0.jpg)](https://www.youtube.com/watch?v=TZbwB7Oh-zw)

# The solution
My work is divided into 5 different notebooks:

1) **Creating the dataset.** We first download the data and create a pandas DataFrame, then we convert this using the Pytorch Geometric "Dataset" interface, implementing an index operator ``dataset[i]`` that processes the DataFrame in order to give us the (processed) data related to a single SKU, i.e. a single instance. Ultimately, I ended up adding a "precompute mode" to the dataset, so we can instead save this processed data to memory as .pt files and simply load them all at once during the training of the LSTM. This was enough to give me maximum training speed when training LSTM models, though it may take a while to load the data into memory.

2) **Training LSTM models**. You can find the code for the models I used (1d CNN, LSTM, Attention Encoder-Decoder) in the notebook. In practice, I trained around 10 total neural  models, tweaking a few parameters here or there. The gain from training multiple models of the same type seemed to be minimal, though. Even LSTM and Attention seemed to perform almost equally and didn't improve each other much when ensembling.

3) **Light Gradient Boosting Model**. In which I create YET ANOTHER dataset with features to be used by the LightGBM model.

4) **Ensemble**. It's just a basic blending, which takes a convex combination of each csv prediction dataframe (For all models, the code saves .csvs for validation and test).

5) **Post-Processing**. If you look at the output given by the ensemble, you can see that the average predicted probability (before summation) decays over the 30 days. According to MercadoLibre's workshop, the uniform distribution was already a decent solution, which motivated me to make my solution more like it. This is accomplished with a simple multiplier trick, and using this prior knowledge worked wonders on the LB. It's obviously a bit risky, but not taking the gamble seemed riskier to me. We'll see where that gets me ;) .


# **Some observations**:

1. **Validation split.** Analyzing the data, it is straightforward to see that the amount sold goes down during the weekends and rises at the start of the week. In order to "align" with the test scenario (starts 1st April - Thursday), my training procedure is such that the model has to look at the data up to March 24th (Wednesday) and predicts how long till stock runs out, starting from the 25th. This validation split produced the best results.

2. **Some of my (public) LB scores**. My first (just a few iterations) LSTM got around 3.93 RPS error, after some more work I arrived at 3.83. Using all the extra models such as lightgbm, I managed to decrease it to 3.7611. After post-processing, we end up with 3.71799.

