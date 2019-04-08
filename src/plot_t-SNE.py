import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import os
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# #     ## Oulu bottleneck layer features logits_0 4/10 folds
# file_expr_pretrained = '/data/zming/logs/cavi/20181113-223830/features_t-sne.npy'
# file_verif_pretrained = '/data/zming/logs/expression/20181113-231429/features_verif_t-sne.npy'
# label_pretrained = '/data/zming/logs/expression/20181113-223830/labels_t-sne.npy'
# #label_verif_pretrained = '/data/zming/logs/expression/20181113-231429/labels_verif_t-sne.npy'
#
# file_expr_singletask = '/data/zming/logs/expression/20181113-224712/features_t-sne.npy'
# file_verif_singletask = '/data/zming/logs/expression/20181113-231224/features_verif_t-sne.npy'
# label_singletask = '/data/zming/logs/expression/20181113-224712/labels_t-sne.npy'
# #label_verif_singletask = '/data/zming/logs/expression/20181113-231224/labels_verif_t-sne.npy'
#
# file_expr_dynmlt = '/data/zming/logs/expression/20181113-225808/features_t-sne.npy'
# file_verif_dynmlt = '/data/zming/logs/expression/20181113-231008/features_verif_t-sne.npy'
# label_dynmlt = '/data/zming/logs/expression/20181113-225808/labels_t-sne.npy'
# #label_verif_dynmlt = '/data/zming/logs/expression/20181113-231008/labels_verif_t-sne.npy'

# label_verif = '../data/IdentitySplit_4th_10fold_oulucasiapairs_Six.txt'

#folder = '20190407-031521'
# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_emb_visual_real.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/label_id_emb_visual_real.npy'

# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_emb_c2v.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/label_id_emb_c2v.npy'

folder = '20190407-031924'

# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_emb_cari_real.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/label_id_emb_cari_real.npy'

file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_emb_v2c.npy'
file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/label_id_emb_v2c.npy'

# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_cari.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/features_cari_label.npy'

# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_cari2visual.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/features_cari2visual_label.npy'

# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_visual2cari.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/features_visual2cari_label.npy'

# file_feature = '/data/zming/logs/cavi/'+'%s'%folder+'/features_cari-visual-verif_emb.npy'
# file_label = '/data/zming/logs/cavi/'+'%s'%folder+'/features_cari-visual-verif_label.npy'

def get_id_label(file):

    id_labels = []
    pairs = []
    paths = []
    with open(file, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)

        for pair in pairs:
            # [cnt, img, img_ref, issame, expr_actual, expr_require, expr_ref, expre_isrequire] = str.split(pair)
            img = pair[1]
            img_ref = pair[2]
            paths.append(img)
            paths.append(img_ref)

        for path in paths:
            id = path.split('/')[6]
            id=int(id[1:])
            id_labels.append(id)

    return id_labels

def main():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    features = file_feature
    labels = file_label
    # digits = datasets.load_digits(n_class=6) ## load 1083 digitals from 0 to 5
    # X, y = digits.data, digits.target
    # n_samples, n_features = X.shape
    X = np.load(features)
    Y = np.load(labels)
    #y = np.load(labels)
    #y = get_id_label(labels)
    #y = y[0:np.shape(X)[0]]

    #if using y as the color, y should be in the range [0-1]
    nclass = 50
    ## mapping labels to integrate
    y_labels_set = list(set(Y))
    y_labels_set.sort()
    y_labels = [y_labels_set.index(x) for x in Y]
    c = Counter(y_labels)
    c_most = c.most_common(nclass)
    labels_most_classes = [x[0] for x in c_most]

    y_max = max(y_labels)
    y_color = []
    X_select = []
    for i, y in enumerate(y_labels):
        if y in labels_most_classes:
            x = y*1.0/y_max
            y_color.append(x)
            X_select.append(X[i])


    # n = 20 ## read 20x20 figures in X
    # img = np.zeros((10 * n, 10 * n))
    # for i in range(n):
    #     ix = 10 * i + 1
    #     for j in range(n):
    #         iy = 10 * j + 1
    #         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
    # plt.figure(figsize=(8, 8)) ##figure size in inches
    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, perplexity=50)
    #tsne = manifold.TSNE(n_components=3, init='pca', random_state=501, perplexity=50)
    X_tsne = tsne.fit_transform(X_select)

    print("Original data dimension is {}. Embedded data dimension is {} ".format(X.shape[-1], X_tsne.shape[-1]))

    '''visualistion of embeeding '''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    ''' normalisation in [0,1] since plt.text can only plot in [0,1] in each axe.'''
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # x = X_norm[:, 0]
    # y = X_norm[:, 1]
    # N = len(y)
    # colors = y
    # area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
    #
    # plt.scatter(x, y, c=colors, alpha=0.5)

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=np.array(y_color), cmap=plt.cm.Spectral)

    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], c=np.array(y_color), cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # plt.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 1], c=np.array(y_color))
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
    #     #          fontdict={'weight': 'bold', 'size': 12})
    #     #plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]))
    #     #plt.scatter(X_norm[i, 0], X_norm[i, 1], c=y[i], cmap=plt.cm.Spectral)
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], c=y_color[i])

    # plt.text(40, -40, str(y[i]), color=plt.cm.Set1(y[i]),
    #          fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

    fig.savefig(os.path.join('/data/zming/logs/cavi/', 't-sne.png'))

    # N = 50
    # x1 = np.random.rand(N)
    # y1 = np.random.rand(N)
    # colors1 = np.random.rand(N)
    #
    # x = X_norm[:, 0]
    # y = X_norm[:, 1]
    # colors = np.array(y_color)
    # area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
    #
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    # plt.show()

if __name__ == '__main__':
    main()

