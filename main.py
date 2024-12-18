import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
import seaborn
seaborn.set(style='whitegrid',font_scale=1.5)

plt.rcParams['font.sans-serif'] = ['Times New Roman']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
plt.rcParams['font.weight'] = 'bold'

REGRESSOR = 'LGB'

def plot(crystal,data_D,label,wavelength,brate,grate):
    dict_k = Counter(label)
    print('dict_k=', dict_k)
    print('score=',dict_k['Q']/len(label))
    plt.figure()
    seaborn.kdeplot(brate,color='b',fill=True,label='$M_B$')
    seaborn.kdeplot(grate,color='g',fill=True,label='$M_G$')
    plt.xlabel('M',fontweight='bold')
    plt.ylabel('Density',fontweight='bold')
    #plt.yticks([])
    plt.legend()
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(crystal+'-kde.png')
    plt.show()
    lr = np.where(label=='Q')[0]
    lg = np.where(label=='G1')[0]
    lgg = np.where(label=='G2')[0]
    lb = np.where(label=='B')[0]
    lbg = np.where(label=='BG1')[0]
    lbgg = np.where(label=='BG2')[0]
    dg = seaborn.xkcd_rgb['teal']
    dy = seaborn.xkcd_rgb['olive']
    plt.figure()
    plt.scatter(brate[lr],grate[lr],color='r',label='Q')
    plt.scatter(brate[lg],grate[lg],color='g',label='G1')
    plt.scatter(brate[lgg],grate[lgg],color=dg,label='G2')
    plt.scatter(brate[lb],grate[lb],color='b',label='B')
    plt.scatter(brate[lbg],grate[lbg],color='y',label='BG1')
    plt.scatter(brate[lbgg],grate[lbgg],color=dy,label='BG2')
    #plt.legend()
    plt.xlabel('$M_B$')
    plt.ylabel('$M_G$')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(crystal+'-scatter.png')
    plt.show()
    rm = data_D[lr].mean(axis=0)
    gm = data_D[lg].mean(axis=0)
    ggm = data_D[lgg].mean(axis=0)
    bm = data_D[lb].mean(axis=0)
    bgm = data_D[lbg].mean(axis=0)
    bggm = data_D[lgg].mean(axis=0)
    plt.figure()
    plt.plot(wavelength,rm,color='r',label='Q')
    plt.plot(wavelength,gm,color='g',label='G1')
    plt.plot(wavelength,ggm,color=dg,label='G2')
    plt.plot(wavelength,bm,color='b',label='B')
    plt.plot(wavelength,bgm,color='y',label='BG1')
    plt.plot(wavelength,bggm,color=dy,label='BG2')
    plt.xlim(wavelength[0],wavelength[-1])      
    plt.legend()
    plt.xlabel('Wavelength (nm)',fontweight='bold')
    plt.ylabel('Intensity (a.u.)',fontweight='bold')
    plt.tight_layout()
    plt.savefig(crystal+'-all.png')
    plt.show()
    plt.figure()
    plt.plot(wavelength[:40],rm[:40],color='r',label='Q')
    plt.plot(wavelength[:40],gm[:40],color='g',label='G1')
    plt.plot(wavelength[:40],ggm[:40],color=dg,label='G2')
    plt.plot(wavelength[:40],bm[:40],color='b',label='B')
    plt.plot(wavelength[:40],bgm[:40],color='y',label='BG1')
    plt.plot(wavelength[:40],bggm[:40],color=dy,label='BG2')
    plt.xlim(wavelength[0],wavelength[39])       
    plt.legend()
    plt.xlabel('Wavelength (nm)',fontweight='bold')
    plt.ylabel('Intensity (a.u.)',fontweight='bold')
    plt.tight_layout()
    plt.savefig(crystal+'-bg.png')
    plt.show()

wavelength = np.loadtxt('wl.txt')
data_D = np.vstack((
    np.loadtxt('d1.txt'),
    np.loadtxt('d2.txt'),
    np.loadtxt('d3.txt'),
    np.loadtxt('d4.txt'),
    np.loadtxt('d5.txt')
))
data_R = np.loadtxt('r.txt')
data_G = np.loadtxt('g.txt')
data_B = np.loadtxt('b.txt')
label = np.loadtxt('l.txt',dtype='<U3')
celsius = ['180','195','210','225','240']
hue = [celsius[i//40000] for i in range(200000)]
brate = data_B/data_R
grate = data_G/data_R
plt.figure()
seaborn.scatterplot(brate,grate,hue=hue,palette=seaborn.color_palette('hls',5))
plt.xlabel('$M_B$')
plt.ylabel('$M_G$')
plt.tight_layout()
plt.show()

plt.figure()
seaborn.boxplot(
    x=np.hstack((hue,hue,hue)),
    y=np.log10(np.hstack((data_B,data_G,data_R))),
    hue=np.hstack((
        ['lg$S_B$' for _ in range(200000)],
        ['lg$S_G$' for _ in range(200000)],
        ['lg$S_R$' for _ in range(200000)]
    )),
    orient='v',
    fliersize=0,
    width=0.5,
    palette=['b','g','r']
)
plt.xlabel('Temperature (${}^\circ C$)',fontweight='bold')
plt.ylabel('lgS',fontweight='bold')
plt.legend(bbox_to_anchor=(1.05,1),loc='best')
plt.tight_layout()
plt.show()

plot('ALL',data_D,label,wavelength,brate,grate)

labeltxt = ['Q','G1','G2','B','BG1','BG2']
palette = ['r','g',seaborn.xkcd_rgb['teal'],'b','y',seaborn.xkcd_rgb['olive']]
y = []
for i in range(len(label)):
    y.append(labeltxt.index(label[i]))

y = np.array(y)

def classifier(xt,yt,xv,yv,i):    
    if REGRESSOR == 'SVM':
        clf = SVC(C=2)
    elif REGRESSOR == 'DT':
        clf = DecisionTreeClassifier(max_depth=5,random_state=42)
    elif REGRESSOR == 'RF':
        clf = RandomForestClassifier(max_depth=5,n_estimators=50,random_state=42,n_jobs=-1)
    elif REGRESSOR == 'LGB':
        clf = lgb.LGBMClassifier(max_depth=5,n_estimators=50,random_state=42)
    else:
        clf = xgb.XGBClassifier(max_depth=5,n_estimators=50,random_state=42)

    clf.fit(xt,yt)
    predict = clf.predict(xv)
    return predict

def evaluate(y,pred,name):
    confusion = confusion_matrix(y,pred)
    acc = accuracy_score(y,pred)
    print(acc)
    plt.figure()
    # plt.title('%s, accuracy=%.3f' % (name,acc))
    seaborn.heatmap(confusion,annot=True,cbar=False,fmt='d',
        xticklabels=labeltxt,
        yticklabels=labeltxt)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('%s-conf.png' % name)
    plt.show()

def kfold(x,y):
    kf = StratifiedKFold(n_splits=5,shuffle=True)
    predict = np.zeros(len(y))
    name = REGRESSOR
    t3 = time.time()
    for i,(train,test) in enumerate(kf.split(x,y)):
        xt = x[train]
        yt = y[train]
        xv = x[test]
        yv = y[test]
        predict[test] = classifier(xt,yt,xv,yv,i)
    
    t4 = time.time()
    print(t4-t3)
    evaluate(y,predict,name)

kfold(data_D,y)