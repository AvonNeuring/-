import pandas as pd

data = pd.read_table('parkinsons_updrs.data')  # 加载本地数据
print(data)  # 打印数据

data.to_csv('machine.csv',sep='|',index=False)  # data转成csv

# 读取生成的machine.csv文件进行验证
csv = pd.read_csv('machine.csv')
print(csv.head(15)) # 打印前15行
import pandas as pd
data = pd.read_csv('machine.csv')
data.head()
data_label = data['total_UPDRS']
data_feature = data.drop(['total_UPDRS'],axis=1)
from sklearn.gaussian_process import GaussianProcessRegressor
gpr = GaussianProcessRegressor()
gpr
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK,WhiteKernel as WK
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
mixed_kernel = kernel = CK(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))+WK(noise_level=1)
gpr = GaussianProcessRegressor(alpha=5,n_restarts_optimizer=20,kernel = mixed_kernel)
X_train, X_test, y_train, y_test = train_test_split(data_feature, data_label, random_state=0)
gpr.fit(X_train,y_train)
test_preds = gpr.predict(X_test)
r2s = r2_score(y_test,test_preds)
print(r2s)
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_test[0:50]['DFA'], test_preds[0:50])
ax.scatter(X_test[0:50]['DFA'],y_test[0:50])
fig.savefig("parkinson.jpg")
from sklearn.externals import joblib
joblib.dump(gpr,"ParkinsonModel.pkl")
