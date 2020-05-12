import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc


def clean(path):
    f=open(path) 
    pcnd=pd.read_csv(f)

    pd.set_option('display.max_columns', 500)

    n_df = pcnd.shape[0] #shape的功能是查看矩阵或者数组的维度
    print ("共有{row}行{col}列数据".format(
            row=pcnd.shape[0], #所以这里就是看pcnd的行数
            col=pcnd.shape[1]))

    colt = pcnd.columns.tolist() #tolist的功能是将数组或者矩阵转换成列表

    tnum = []
    tmis = []
    t = 1
    prob = 0.8

    for col in colt:
        missing = n_df - np.count_nonzero(pcnd[col].isnull().values)
        mis_perc = 100 - float(missing) / n_df * 100 #float就是将数据改成小数点形式。eg：10 -> 10.0
        print ("{col}的缺失比例是{miss}%".format(col=col,miss=mis_perc))
        tnum.append(t) #用于在列表末尾添加新的对象
        tmis.append(mis_perc)
        t = t + 1
        if mis_perc > prob :
            pcnd.drop([col],axis=1,inplace=True)
            
    cols = pcnd.columns.tolist()
            
    cat_tran = []
    print ("\nDescriptive variables有:")
    for col in cols:
        if pcnd[col].dtype == "object":
            print (col)
            cat_tran.append(col)
                    
    print ("\n开始转换Descriptive variables...")
    le = preprocessing.LabelEncoder()
    for col in cat_tran:
        tran1 = le.fit_transform(pcnd[col].tolist())
        tran_df = pd.DataFrame(tran1, columns=['num_'+col])
        print("{col}经过转化为{num_col}".format(col=col,num_col='num_'+col))
        #把descriptive variables从原本的dataset中删掉，再把转换过的加回去
        pcnd.drop([col],axis=1,inplace=True)
        pcnd = pd.concat([pcnd, tran_df], axis=1)
    print('')
    return pcnd

#整个上面一大段都是在定义clean这个公式，下面我们才开始用
print('data')

data=clean('D:/Academics/Past Semesters/Fall 1/IEEE/sum.csv')
data.head()

data = data.fillna(999) 
data1 = data.drop(['num_ProductCD', 'num_card4', 'Unnamed: 0', 'num_card6', 'isFraud'], axis=1)
name = data1.keys()

# 分training and test datasets:
X_train = data1.drop(['TransactionID','TransactionDT'], axis=1)
Y_train = data['isFraud']

#---------------------------------------------------------------------------------------------

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=1995,
    tree_method='gpu_hist',
    objective='binary:logistic',
)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, Y_train, test_size=0.5)
y_score = clf.fit(X_train1, y_train1).predict(X_test1)
fpr,tpr,threshold = roc_curve(y_test1, y_score)

auc_score = auc(fpr,tpr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='LR (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

