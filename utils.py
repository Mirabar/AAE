import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, MeanSquaredError
from tensorflow.keras.activations import softmax
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

def e_loss(true, pred, n_classes=10, bc=10, bn=10, combined_loss=True):
    '''
    :param true: originally sampled latent vector
    :param pred: encoder reconstructed latent vector
    :param n_classes: number of classes
    :param bc: zc regularization coefficient
    :param bn: zc regularization coefficient
    :param combined_loss: boolean for whether to joint tzn and zc losses to one
    :return: float for encoder loss of list of floats for zc and zn loss
    '''

    zn = true[0]
    zc = true[1]
    ezn = pred[:, :-n_classes]
    ezc = pred[:, -n_classes:]

    zc_logits = softmax(ezc)
    cel = tf.reduce_mean(categorical_crossentropy(zc, zc_logits))
    msel = MeanSquaredError()(zn, ezn)
    if combined_loss:
        loss = bc * cel + bn * msel
    else:
        loss = [cel, msel]

    return loss


def z_sampler(batch_size, n_classes=10, zn_dim=30, sigma=0.1, zc_ind=None, concat=False):
    '''
    :param batch_size: integer for number of samples in mini-batch
    :param n_classes: integer for number of classes
    :param zn_dim: noise dimenssion
    :param sigma: float for noise std
    :param zc_ind: if not None, zc will be selected for the c_ind index
    :param concat: boolean for whether to concat zc and zn
    :return: 2-D of shape batch_size-by-(zn_dim+n_classes) or two arrays for zc and zn
    '''

    if zc_ind is None:
        zc_ind = np.random.randint(0, n_classes, size=(batch_size,))

    zc = np.zeros((batch_size, n_classes))

    zc[range(batch_size), zc_ind] = 1

    zn = sigma * np.random.randn(batch_size, zn_dim)

    if concat:
        return np.hstack((zn, zc))
    else:
        return tf.convert_to_tensor(zn, dtype=tf.float32), tf.convert_to_tensor(zc, dtype=tf.float32)


def cluster_viz(latent, labels, save_dir, n_classes=10, fig_name='tsne_viz.png'):
    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param save_dir: string for destination directory
    :param n_classes: integer for number of classes
    :param fig_name: string for figure name
    '''
    colors = cm.rainbow(np.linspace(0, 1, n_classes))

    tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
    tsne_embed = tsne.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(16, 10))
    for i, c in zip(range(n_classes), colors):
        ind_class = np.argwhere(labels == i)
        ax.scatter(tsne_embed[ind_class, 0], tsne_embed[ind_class, 1], color=c, label=i, s=5)
    ax.set_title('t-SNE vizualization of latent vectors')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'{save_dir}/{fig_name}')
    plt.close('all')


def cluster_latent(latent, labels, n_classes=10):
    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param n_classes: integer for number of classes
    :return: evaluation of prediction after applying K-means on the predicted latent vectors
    '''
    km = KMeans(n_clusters=n_classes, random_state=0).fit(latent)
    labels_pred = km.labels_

    acc_c, acc_all = cluster_acc(labels_pred, labels)
    ari = adjusted_rand_score(labels, labels_pred)
    nmi = normalized_mutual_info_score(labels, labels_pred)

    return acc_c, acc_all, ari, nmi


def cluster_acc(pred, true):
    '''
    :param pred: predicted cluster
    :param true: true labels
    :return: classification accuracy for complete data and per class
    '''
    c_dict = {}
    all_pred = 0
    for c in np.unique(true):
        c_ind = np.where(true == c)
        true_c = true[c_ind]
        pred_c = pred[c_ind]
        pred_c_unique = np.unique(pred_c, return_counts=True)
        pred_c_max = pred_c_unique[0][pred_c_unique[1].argmax()]
        num_pred_c = pred_c_unique[1].max()
        all_pred += num_pred_c

        c_dict[str(c)] = [str(pred_c_max), num_pred_c/len(true_c)]

    return c_dict, all_pred/len(true)

