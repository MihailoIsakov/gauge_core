import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import torch
from joblib import Memory
import logging
import xgboost as xgb


import autoencoder


memory = Memory(".joblib_cache")


def get_DBSCAN_labels(df, eps, metric='euclidean'):
    logging.info("Running DBSCAN on {} points".format(df.shape[0]))
    return DBSCAN(eps=eps, min_samples=10, metric=metric, n_jobs=8).fit(df).labels_


def tsne2(data, perplexity=50):
    return TSNE(n_components=2, perplexity=perplexity, random_state=10).fit_transform(data)


def build_lowdim_data(df_scaled, epochs=100):
    """
    Train an autoencoder on the scaled data and project it to the latent dimension
    """
    device       = "cuda"
    batch_size   = 2**10
    lr           = 1e-3
    test_ratio   = 0.2
    encoder_size = [df_scaled.shape[1], 30, 10]
    decoder_size = [10, 30, df_scaled.shape[1]]

    X_train, X_test = sklearn.model_selection.train_test_split(np.array(df_scaled).astype(np.float32), test_size=test_ratio)
    X_train, X_test = torch.Tensor(X_train).to(device), torch.Tensor(X_test).to(device)

    def train_AE(X_train, X_test, encoder_size, decoder_size, epochs, lr):
        model = autoencoder.AutoEncoder(encoder_size, decoder_size).to(device)
        autoencoder.train(model, dataset=X_train, num_epochs=epochs, batch_size=batch_size, lr=lr)

        # Calculate training error
        AE_reconstruction = model(X_train)
        training_error = np.mean(((X_train - AE_reconstruction[0])**2).cpu().detach().numpy())

        # Calculate test error
        AE_reconstruction = model(X_test)
        test_error = np.mean(((X_test - AE_reconstruction[0])**2).cpu().detach().numpy())

        return model, training_error, test_error

    model, training_error, test_error = train_AE(X_train, X_test, encoder_size, decoder_size, epochs, lr)
    logging.info("Training mean L2 loss: {}, test mean L2 loss: {}".format(training_error, test_error))

    # Let's map the original dataset and run HiPlot on it
    encoder = list(model.children())[0]
    lowdim_data = encoder(torch.Tensor(np.array(df_scaled).astype(np.float32)).to(device))
    lowdim_data = lowdim_data.cpu().detach().numpy()

    return lowdim_data


def train_model(X, y, model):
    """
    Partitions the dataset into the training and test split, trains a model specified by the model string,
    and returns the test set, predictions, and the model.
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    # Train linear regression
    if model == "linear":
        regressor = sklearn.linear_model.LinearRegression()
    elif model == "ridge": 
        regressor = sklearn.linear_model.Ridge(alpha=0.1)
    elif model == "lasso": 
        regressor = sklearn.linear_model.Lasso(alpha=0.01)
    elif model == "elastic": 
        regressor = sklearn.linear_model.ElasticNet(alpha=0.01, l1_ratio=0.1)
    elif model == "decision_tree":
        regressor = DecisionTreeRegressor(max_depth=5)
    elif model == "gbm":
        regressor = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                     max_depth=5,            alpha=10,             n_estimators=10)

    regressor.fit(X_train, y_train)

    # Predict and plot
    y_pred = regressor.predict(X_test)

    return y_test, y_pred, regressor, (X_train, X_test, y_train, y_test)
