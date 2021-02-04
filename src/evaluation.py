from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def confusion(pred_df, labels, dst, title=None, size=5):
    conf = confusion_matrix(pred_df.label, pred_df.pred, labels, normalize='true')
    conf = np.round(conf, decimals=2)
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.imshow(conf, cmap='viridis', vmax=2)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel('Label')
    ax.set_xlabel('Prediction')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, conf[i, j],
                        ha="center", va="center", color="w")
    if dst is not None:
        plt.savefig(dst, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def failures(pred_df, dst, n_max=100):
    fails = pred_df[pred_df.label != pred_df.pred]
    fails = fails[['label', 'pred', 'text']]
    n_max = min(n_max, len(fails.index))
    fails.sample(n_max).to_csv(dst)