"""
Created on Fri Apr 13 00:30:19 2018

@author: chander
"""

import skimage
import numpy as np
import pywt
import scipy.stats
import cpbd
from matplotlib import pyplot as plt
plt.ion()
import glob
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from scipy import io 
import itertools
from scipy import interp

def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def effective_mean(patch):
    vec=patch.flatten()
    m1 = [i for i in vec if i!=0]
    return(np.mean(m1))

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def ReturnPixelStats(img):
    f = []
    #Pixel Statistics
    for channel in range(0,3):
        a = img[:,:,channel]
        vec = a.flatten()
        # 8 statistic characteristics of pixel values for each color
        f.append(np.mean(vec))
        f.append(np.var(vec))
        f.append(scipy.stats.skew(vec))
        f.append(scipy.stats.kurtosis(vec))
        f.append(np.percentile(vec,25))
        f.append(np.median(vec))
        f.append(np.percentile(vec,75))
        f.append(scipy.stats.entropy(vec))
    return f
    
def ExtractTexture(img):
    glcm = skimage.feature.greycomatrix(img,[1],[0, np.pi/4, np.pi/2, 3*np.pi/4],normed=True, symmetric=True)
    contrast = skimage.feature.greycoprops(glcm, prop = 'contrast')
    homogeneity = skimage.feature.greycoprops(glcm, prop = 'homogeneity')
    dissimilarity = skimage.feature.greycoprops(glcm, prop = 'dissimilarity')
    ASM = skimage.feature.greycoprops(glcm, prop = 'ASM')
    correlation = skimage.feature.greycoprops(glcm, prop = 'correlation')
    energy = skimage.feature.greycoprops(glcm, prop = 'energy')
    features = (np.array([contrast, homogeneity, dissimilarity, ASM, correlation, energy])).flatten()
    return features


def get_wavelet_features_gray(img):
    wp = pywt.WaveletPacket2D(data=img, wavelet='haar', mode='symmetric')
    feature=[]
    data=wp['h'].data
    vec=data.flatten()
    feature.append(ReturnPixelStats(vec))

    data=wp['v'].data
    vec=data.flatten()
    feature.append(ReturnPixelStats(vec))

    data=wp['d'].data
    vec=data.flatten()
    feature.append(ReturnPixelStats(vec))

    return feature

def GetImageList(path1, path2):
    g_img,b_img=[],[]
    
    for i in sorted(glob.glob(path1+'*.JPG')):
        g_img.append(i)

    for i in sorted(glob.glob(path2+'*.JPG')):
        b_img.append(i)
    
    return(g_img,b_img)

def GetFeatures(g_img, b_img):
    GoodFeatures = []
    count = 1
    for img_loc in g_img:
        print('Processing Image ', count)
        count = count + 1
        img = plt.imread(img_loc)
        GoodFeatures.append(ImageFeatures(img))
        
    BadFeatures = []
    for img_loc in b_img:
        print('Processing Image ', count)
        count = count + 1
        img = plt.imread(img_loc)
        BadFeatures.append(ImageFeatures(img))
        
    return (GoodFeatures, BadFeatures)


def ImageFeatures(img):
    (h,l) = img[:,:,0].shape
    #Initialize feature vector to 0
    f = []
    
    #Pixel Statistics
    f.extend(ReturnPixelStats(img))

    #Texture Features
    #Function Takes uint8 greyscale image as input
    grey_img = rgb2gray(img)
    a_8 = grey_img.astype(np.uint8)
    f.extend(ExtractTexture(a_8))
    
    #Extracting Region Specific Features
    a = grey_img
    A1 = a[:int(h/4),:int(l/4)]
    A2 = a[:int(h/4), int(5*l/24):int(5*l/24+l/4)]
    A3 = a[:int(h/4):,int(3*l/4):]
    A4 = a[int(5*h/24):int(5*h/24+h/4),:int(l/4)]
    A6 = a[int(5*h/24): int(5*h/24+h/4),int(l-l/4):]
    A7 = a[-int(h/4):,:int(l/4)]
    A8 = a[int(h-h/4):,int(5*l/24):int(5*l/24+l/4)]
    A9 = a[int(h-h/4):, int(l-l/4):]
    #Central Pixel Statistics
    A5_color = img[int(5*h/24):int(5*h/24+h/4), int(5*l/24):int(5*l/24+l/4),:]
    f.extend(ReturnPixelStats(A5_color))
    
    #features to define symmetricity of the fundus image
    m1=effective_mean(A1)
    m2=effective_mean(A2)
    m3=effective_mean(A3)
    m4=effective_mean(A4)
    m6=effective_mean(A6)
    m7=effective_mean(A7)
    m8=effective_mean(A8)
    m9=effective_mean(A9)
    d1 = m1-m9
    d2 = m2-m8
    d3 = m3-m7
    d4 = m4-m6
    f.append(d1)
    f.append(d2)
    f.append(d3)
    f.append(d4)

    w=[]

    #wavelet features
    w.extend(get_wavelet_features_gray(img[:,:,0]))
    w.extend(get_wavelet_features_gray(img[:,:,1]))
    w.extend(get_wavelet_features_gray(img[:,:,2]))
    f.extend(w)
    
    #CPBD Blur Feature
    f.append(cpbd.compute(grey_img))
    return f



#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
class_names = np.asarray(['Gradable','Non-Gradable'])
from sklearn import svm    

#kfold = StratifiedKFold(n_splits=5, shuffle=True)

r_kfold = RepeatedKFold(n_splits=5, n_repeats=100)
kfold = r_kfold



def ClassifyPLSLDA(data_list, target_list):
    data = np.concatenate(data_list, axis=0)
    targets = np.concatenate(target_list, axis=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    conf_matrices = []
    targets = targets[:,1]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in kfold.split(data, targets):
        #PLS followed by LDA
        pls_transform = PLSRegression(n_components = 10)
        pls_transform.fit_transform(X = data[train], y = targets[train])
        train_X_transformed = pls_transform.transform(data[train])
        
        clf = LinearDiscriminantAnalysis()
        clf.fit(train_X_transformed, targets[train])
        pred_labels = clf.predict(pls_transform.transform(data[test]))
        true_labels = targets[test]
        
        (true_positive, true_negative, false_positive, false_negative) = (0.0,0.0,0.0,0.0)
        
        for i in range(0,len(pred_labels)):
            if (pred_labels[i], true_labels[i]) == (0,0):
                true_positive = true_positive + 1
            if (pred_labels[i], true_labels[i]) == (1,1):
                true_negative = true_negative + 1
            if (pred_labels[i], true_labels[i]) == (0,1):
                false_positive = false_positive + 1
            if (pred_labels[i], true_labels[i]) == (1,0):
                false_negative = false_negative + 1
        
        detection_rate = true_positive/(true_positive + false_negative)
        false_alarm = false_positive/(true_negative + false_positive)
        conf_matrix = np.array([[detection_rate, 1 - detection_rate],[false_alarm, 1 - false_alarm]])
        conf_matrices.append(conf_matrix)
        
        preds = clf.predict_proba(pls_transform.transform(data[test]))[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(targets[test], preds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()       
    #    plt.figure()
#    roc_auc = metrics.auc(fpr, tpr)
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#    plt.legend(loc = 'lower right')
#    plt.plot([0, 1], [0, 1],'r--')
#    plt.xlim([0, 1])
#    plt.ylim([0, 1])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show()
#        
    conf = np.array(conf_matrices)
    mean_conf_matrix = np.mean(conf, axis = 0)
    var_conf_matrix = np.var(conf, axis = 0)
    acc = (mean_conf_matrix[0,0]+mean_conf_matrix[1,1])/2.0
    acc_var = (var_conf_matrix[0,0]+var_conf_matrix[1,1])/2.0
    plot_confusion_matrix(mean_conf_matrix, classes=class_names,title='Average Confusion matrix PLS-LDA,accuracy= %0.2f '%acc)
    

    
        
    return mean_conf_matrix,var_conf_matrix,acc,acc_var

def ClassifySVM(data_list, target_list):
    data = np.concatenate(data_list, axis=0)
    targets = np.concatenate(target_list, axis=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    conf_matrices = []
    Y = targets[:,1]
    X = data
    cvscores,conf_matrices,accuracy = [],[],[]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in kfold.split(X, Y):
        clf = svm.SVC(probability=True)
        clf.fit(X[train],Y[train])
        pred_labels = clf.predict(X[test])
        true_labels = Y[test]
        print("SVM accuracy =",clf.score(X[test],Y[test])) 
        
        (true_positive, true_negative, false_positive, false_negative) = (0.0,0.0,0.0,0.0)
        for i in range(0,len(pred_labels)):
            if (pred_labels[i], true_labels[i]) == (0,0):
                true_positive = true_positive + 1
            if (pred_labels[i], true_labels[i]) == (1,1):
                true_negative = true_negative + 1
            if (pred_labels[i], true_labels[i]) == (0,1):
                false_positive = false_positive + 1
            if (pred_labels[i], true_labels[i]) == (1,0):
                false_negative = false_negative + 1
        detection_rate = true_positive/(true_positive + false_negative)
        false_alarm = false_positive/(true_negative + false_positive)
        acc = (true_positive + true_negative)/(true_positive+false_negative+true_negative+false_positive)
        conf_matrix = np.array([[detection_rate, 1 - detection_rate],[false_alarm, 1 - false_alarm]])
        accuracy.append(acc)
        conf_matrices.append(conf_matrix)
        score=clf.score(X[test],Y[test])
        cvscores.append(score)
        preds = clf.predict_proba(X[test])[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(Y[test], preds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
   
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='black', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()       
    #    plt.fi

    conf_matrices =  np.asarray(conf_matrices)
    accuracy_avg = np.mean(accuracy)
    accuracy_var = np.var(accuracy)
    conf_mat_avg =np.mean(conf_matrices,axis=0)
    conf_mat_var =np.var(conf_matrices,axis=0)
    plt.figure()
    class_names = np.asarray(['Gradable','Non-Gradable'])
    plot_confusion_matrix(conf_mat_avg, classes=class_names,title='Average Confusion matrix,accuracy= %0.2f '%accuracy_avg)
    return conf_mat_avg,conf_mat_var,accuracy_avg,accuracy_var


def ClassifyAdaBoost(data_list, target_list):
    data = np.concatenate(data_list, axis=0)
    targets = np.concatenate(target_list, axis=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    conf_matrices = []
    targets = targets[:,1]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in kfold.split(data, targets):
        train_X = data[train]
        train_Y = targets[train]
        test_X = data[test]
        test_Y = targets[test]
        
        #AdaBoost Decision Tree

      
        adbt_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                                      algorithm= "SAMME",
                                      n_estimators=1000)
        adbt_clf.fit(train_X, train_Y)
        pred_labels = adbt_clf.predict(test_X)
        true_labels = test_Y
        
        (true_positive, true_negative, false_positive, false_negative) = (0.0,0.0,0.0,0.0)
        
        for i in range(0,len(pred_labels)):
            if (pred_labels[i], true_labels[i]) == (0,0):
                true_positive = true_positive + 1
            if (pred_labels[i], true_labels[i]) == (1,1):
                true_negative = true_negative + 1
            if (pred_labels[i], true_labels[i]) == (0,1):
                false_positive = false_positive + 1
            if (pred_labels[i], true_labels[i]) == (1,0):
                false_negative = false_negative + 1
        
        detection_rate = true_positive/(true_positive + false_negative)
        false_alarm = false_positive/(true_negative + false_positive)
        conf_matrix = np.array([[detection_rate, 1 - detection_rate],[false_alarm, 1 - false_alarm]])
        conf_matrices.append(conf_matrix)
        
        preds = adbt_clf.predict_proba(test_X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(targets[test], preds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
   
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    conf = np.array(conf_matrices)
    mean_conf_matrix = np.mean(conf, axis = 0)
    var_conf_matrix = np.var(conf, axis = 0)
    acc = (mean_conf_matrix[0,0]+mean_conf_matrix[1,1])/2.0
    acc_var = (var_conf_matrix[0,0]+var_conf_matrix[1,1])/2.0
    plot_confusion_matrix(mean_conf_matrix, classes=class_names,title='Average Confusion matrix Adaboost, accuracy= %0.2f '%acc)    
    return mean_conf_matrix,var_conf_matrix,acc,acc_var


good_dir = '/home/chander/Documents/study_material/image_work/dataset/GoodQSphoto/'
bad_dir = '/home/chander/Documents/study_material/image_work/dataset/BadQSphoto/'

#extract the features

#(g_img,b_img) = GetImageList(good_dir, bad_dir)
#g_img,b_img=g_img[0],b_img[0]
#(good_features, bad_features) = GetFeatures(g_img, b_img)
##save the features
#io.savemat('good_features_new.mat',{'a':good_features})
#io.savemat('bad_features_new.mat',{'a':bad_features})
#load features

good_loc = '/home/chander/Documents/study_material/image_work/IQA/good_features_new.mat'
bad_loc = '/home/chander/Documents/study_material/image_work/IQA/bad_features_new.mat'
good_features =io.loadmat(good_loc)
good_features = good_features['a']
bad_features = io.loadmat(bad_loc)   
bad_features = bad_features['a']


target_1 = np.array([[1,0] for i in range(0,len(good_features))])
target_2 = np.array([[0,1] for i in range(0,len(bad_features))])
conf_svm = ClassifySVM([good_features, bad_features], [target_1, target_2])
conf_PLSLDA = ClassifyPLSLDA([good_features, bad_features], [target_1, target_2])
conf_adaboost = ClassifyAdaBoost([good_features, bad_features], [target_1, target_2])


