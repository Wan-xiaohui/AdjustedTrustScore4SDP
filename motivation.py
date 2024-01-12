import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 导入分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap

# 定义分类器名称和实例
names = ["NB", "DT", "RF", "LR", "SVM", "MLP"]
classifiers = [
    GaussianNB(),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    LogisticRegression(random_state=42),
    SVC(probability=True, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42)
]

# 创建一个更难的二分类数据集
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, flip_y=0.2, class_sep=1.0, random_state=42)

# 定义颜色
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# 重新生成新样本
np.random.seed(0)  # 确保样本是可复现的
new_sample = np.random.uniform(X.min(axis=0), X.max(axis=0), (1, 2))

# 重新绘制分类边界和预测概率
plt.figure(figsize=(18, 3))
for index, (name, classifier) in enumerate(zip(names, classifiers)):
    ax = plt.subplot(1, len(classifiers), index + 1)
    classifier = make_pipeline(StandardScaler(), classifier)
    classifier.fit(X, y)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    proba = classifier.predict_proba(new_sample)[0]
    proba_text = f'Confidence: [{proba[0]:.2f}, {proba[1]:.2f}]'

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
    ax.scatter(new_sample[:, 0], new_sample[:, 1], marker='x', c='green', s=100)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    # ax.set_title(name)

    plt.text(0.03, 0.95, name, transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.03, 0.03, proba_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom')

plt.tight_layout()
plt.savefig('pred_prob.png')
plt.show()

