import pandas as pd 

#讀取CSV
data_path = 'C:\\Users\\Yan_Tse\\Desktop\\Data science\\HeartDisease\\heart.csv'
df_heart=  pd.read_csv(data_path)

#row and column 數量
print(df_heart.shape)
print(df_heart.head())

# #檢查是否缺漏1]
# print(df_heart.isnull().sum())

#分成TraindData and TestData
X, y = df_heart.iloc[ : , :-2].values, df_heart.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y) #stratify　按照　ｙ的比例分配
print(X_train.shape)
print(y_train.shape)


#以隨機森林分類評估特徵的重要性
import numpy as np
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000, random_state=1)
forest.fit(X_train, y_train)
feat_labels = df_heart.columns[1:] # X labels
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]  #由大到小
for f in range(X_train.shape[1]):
    print("%2d %-*s %f"% (f+1, 30,
                        feat_labels[indices[f]],        
                        importances[indices[f]]))

print('\n')
#利用SelectFromModel減少feature數量
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold =0.1, prefit=True)
X_selected = sfm.transform(X_train)
for f in range(X_selected.shape[1]):
    print("%2d %-*s %f"% (f+1, 30,
                        feat_labels[indices[f]],        
                        importances[indices[f]]))  #剩四個特徵


#降維來壓縮數據
#方法一: PCA主成分分析
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    #setup marker generator and colors map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution,))
    
    Z = classifier.predict(np.array([xx1.reveal(), xx2.reveal()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c = cmap(idx), edgecolor = 'black', markers=markers[idx], label=cl)


