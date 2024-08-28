import numpy as np
import torch
import logging

from Model import TimesNet, UnetResidualBiLSTM
from SleepDataset import SleepDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return np.matmul(x, x.T)

def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means)/2 # 對means進行中心化，除以2是因為行列都要減去means
        gram -= means[:, None]
        gram -= means[None, :]

    return gram

def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.

    x 與 y 都會先Compute Gram (kernel) matrix >> x = np.matmul(x, x.T), 變成大小一樣的方陣，且為一個對稱矩陣  >> gram_linear(x)
    因為x, y為symmetric matrix, 因此行列平均相等， 只需要np.mean(x, 0)就可以取得平均值，再對矩陣做中心化，對每個矩陣的行和列進行中心化的目的是確保矩陣的行和列的均值都為零 >> center_gram(gram, unbiased=False)

    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    """ ravel() 將陣列拉成一維向輛 """
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    """ np.linalg.norm 返回矩陣或向量的長度、大小、norm """
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
        The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnetResidualBiLSTM.ArousalApneaUENNModel(size=5*60*100, num_class=1, n_features=8)
    # model = TimesNet.TimesNet(seq_length=5*60*100, num_class=1, n_features=8, layer=3)

    logger = get_logger(fr'weight\Arousal_Apnea\Train_1219\CKASimilarity.log')
    logger.info(f"CKA Calute between timenet encoder and input data")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    model.load_state_dict(torch.load(rf'weight\Arousal_Apnea\Train_1219\model19_1.2054310477436512.pth'))
    model = model.eval()
    logger.info("Successful load model")

    

    Valtrain_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Train"
    Vallabel_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Label"
    allDataset = SleepDataset(rootX = Valtrain_datapath, rooty = Vallabel_datapath,
                      transform=None)
    test_iter = DataLoader(dataset=allDataset,
                        batch_size=8, 
                        drop_last=True,
                        shuffle=False)
    
    x, y = next(iter(test_iter))
    x = x.to(device=device, dtype=torch.float32)
    logger.info(f"input Data shape: {x.shape}")
    arousal, apnea, timesnetOutput = model(x)
    logger.info(f"TimesNet Output Data shape: {timesnetOutput.shape}")

    cka_all = 0
    

    for X, y in tqdm(test_iter):
        X = X.to(device=device, dtype=torch.float32)
        arousal, apnea, timesnetOutput = model(x)
        X = X.detach().cpu().numpy()
        timesnetOutput = timesnetOutput.detach().cpu().numpy()
        for i in range(X.shape[0]):
            cka_from_examples = cka(gram_linear(X[i]), gram_linear(timesnetOutput[i]))
            cka_all += cka_from_examples

    logger.info(f"CKA Similarity: {cka_all/len(allDataset)}")