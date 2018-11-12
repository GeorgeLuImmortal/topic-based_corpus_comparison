
# coding: utf-8

# In[5]:


import re
import numpy as np
from collections import Counter
import operator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing


# In[1]:


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# In[2]:


def one_versus_all(labels,X,df_corpora):

    word_fre_counter={}

    for label in list(set(labels)):
        target_index = np.array(df_corpora[df_corpora['label']==label].index.tolist())
        a = np.ones(X.shape[0],dtype=bool)
        a[target_index] = False 
        

        X_a = X[~a,:]
        X_b = X[a,:]

        counter_1=Counter()
        counter_2=Counter()

        for sample in X_a:
            for idx,value in enumerate(sample):
                if value!=0:
                    counter_1[idx] = counter_1[idx]+value

        for sample in X_b:
            for idx,value in enumerate(sample):
                if value!=0:
                    counter_2[idx] = counter_2[idx]+value

        word_fre_counter[label]=(counter_1,counter_2)
    return word_fre_counter


def one_versus_all_doc(labels,X,df_corpora):

    doc_fre_counter={}
   
    for label in list(set(labels)):
        target_index = np.array(df_corpora[df_corpora['label']==label].index.tolist())
       
        a = np.ones(X.shape[0],dtype=bool)
        a[target_index] = False 
        

        X_a = X[~a,:]
        X_b = X[a,:]
        
        counter_doc_1=Counter()
        counter_doc_2=Counter()

        counter_doc_1.update(np.where(X_a>0)[1])
        counter_doc_2.update(np.where(X_b>0)[1])
        
        doc_fre_counter[label]=(counter_doc_1,counter_doc_2)
        
    return doc_fre_counter


# In[3]:
def ext_jsd(word_fre_counter,leng,model_num):

    rest_part = {}
    m = {}
    ext_jsd_score={}
    
    for key in list(range(model_num)):
        m_numerator = 0
        rest_ = 0
        for item in word_fre_counter.keys():
            m_numerator+= word_fre_counter[item][0][key]
            A = sum(word_fre_counter[item][0].values())
            a = word_fre_counter[item][0][key]
            rest_ += (a/leng)*np.log2(a/A)

        m[key]=m_numerator/leng
        rest_part[key] = rest_
        ext_jsd_score[key] = -(m[key])*np.log2(m[key])+rest_part[key]
        
    return ext_jsd_score

def chi_square(A,B,C,D):
    N = A+B+C+D
    return np.float64(N*(A*D-B*C)*(A*D-B*C)/(A+B)/(C+D)/(A+C)/(B+D))

# def jensen_shannon_divergence(A,B,C,D):
#     N = A+B+C+D
#     return np.float64(-((A+C)/N)*np.log2((A+C)/N)+A/N*np.log2(A/(A+B))+C/N*np.log2(C/(C+D)))

def jensen_shannon_divergence(A,B,C,D):
    N = A+B+C+D
    return np.float64(-((A+C)/N)*np.nan_to_num(np.log2((A+C)/N))+A/N*np.nan_to_num(np.log2(A/(A+B)))+C/N*np.nan_to_num(np.log2(C/(C+D))))

def information_gain(A,B,C,D):
    N = A+B+C+D
    return np.float64(A/N*np.log2(A*N)/(A+B)/(A+C)*B/N*np.log2(B*N)/(A+B)/(B+D)*C/N*np.log2(C*N)/(A+C)/(D+C)*D/N*np.log2(D*N)/(D+B)/(D+C))

def mutual_information(A,B,C,D):
    N = A+B+C+D
    return np.float64(np.log2(A*(A+B)/N/(A+C)))

def gain_ratio(A,B,C,D):
    N = A+B+C+D
    ig = information_gain(A,B,C,D)
    denominator=np.float64((A+B)/N*np.log2((A+B)/N)+(C+D)/N*np.log2((C+D)/N))
    return np.float64(-ig/denominator)

def relevance_frequency(A,C):
    return np.log2(2+A/C)
def inverse_document_frequency(A,C,no_of_doc):

    return np.log2(no_of_doc/(A+C))


# In[4]:

def metrics_doc(method,countera,counterb):
    
    counter_={}
    counter_a = countera.copy()
    counter_b = counterb.copy()
   
    
    for key in counter_a.keys():
        if key not in counter_b.keys():
            counter_b[key]=1e-20
        
    for key in counter_b.keys():
        if key not in counter_a.keys():
            counter_a[key]=1e-20
    
    for key in counter_a.keys():
        
        A = np.float64(counter_a[key])
        C = np.float64(counter_b[key])
        
        
        if method == 'rf':
            rf = relevance_frequency(A,C)

            counter_[key] = rf
        # elif method == 'idf':
        #     idf = inverse_document_frequency(A,C,no_of_doc)

        #     counter_[key] = idf
        
    return counter_

def metrics_update(method,countera,counterb):
    counter_={}
    counter_a = countera.copy()
    counter_b = counterb.copy()
    sum_a = np.float64(sum(counter_a.values()))
    sum_b = np.float64(sum(counter_b.values()))
    
    N=sum_a+sum_b
    
    for key in counter_a.keys():
        if key not in counter_b.keys():
            counter_b[key]=1e-20
        
    for key in counter_b.keys():
        if key not in counter_a.keys():
            counter_a[key]=1e-20
            
        
    for key in counter_a.keys():
        A = np.float64(counter_a[key])
        C = np.float64(counter_b[key])
        
        B = np.float64(sum_a-A)
        D = np.float64(sum_b-C)
        
        
        
        if method == 'chi':
            chi = chi_square(A,B,C,D)
            
            counter_[key] = chi
        elif method == 'mi':
            mi = mutual_information(A,B,C,D)
            
            counter_[key] = mi
                      
        elif method == 'ig':
            ig = information_gain(A,B,C,D)
            
            counter_[key] = ig
                     
            
        elif method == 'gr':
            gr = gain_ratio(A,B,C,D)
            
            counter_[key] = gr
        
        elif method == 'jsd':
            jsd = jensen_shannon_divergence(A,B,C,D)    
            counter_[key] = jsd
            
        elif method == 'rf':
            rf = relevance_frequency(A,C)

            counter_[key] = rf
                   
    return counter_


def grid_search(classifier,random_seed1,random_seed2,eval_metric_list,X,Y):
    # shuffle dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.0, random_state=random_seed1)
    
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X, Y, test_size=0.0, random_state=random_seed2)

    if classifier=='svm':
    # Set the parameters by cross-validation for svm
        tuned_parameters = [ 
                            {'loss': ['hinge'], 'penalty': ['l2'],
                             'C': [0.1,1, 10, 100, 1000],},
                            {'loss': ['squared_hinge'], 'penalty': ['l2'],
                             'C': [0.1,1, 10, 100, 1000]}
                           ]
        
    elif classifier=='nb':
        # Set the parameters by cross-validation for nb
        tuned_parameters = [ 
                            {'fit_prior': [True,False], 'alpha': [1e-10,1e-5,1e-1,0.5,1.0],
                             }
                           ]
        
        
    scores = eval_metric_list
    
    

    for score in scores:
        params_=[]
        scores_=[]
        print("# Tuning hyper-parameters for %s" % score)
        print()
        
        if classifier=='svm':
        #cv for svm
            clf = GridSearchCV(svm.LinearSVC(), tuned_parameters, cv=5,
                           scoring='%s' % score,n_jobs = 2)
        #cv for nb
        elif classifier=='nb':
            clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5,
                               scoring='%s' % score)

        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            params_.append(params)
            scores_.append("%0.5f (+/-%0.03f)"% (mean, std * 2))
        print()
        
        
        
        if classifier=='svm':
            best_parameters=[ 
                        {'loss': [clf.best_params_['loss']], 'penalty': [clf.best_params_['penalty']],
                         'C': [clf.best_params_['C']],}
                       ]
            del clf
            clf = GridSearchCV(svm.LinearSVC(), best_parameters, cv=5,
                               scoring='%s' % score,n_jobs = 2)
            clf.fit(X_train_a, y_train_a)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            print('----------')
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.5f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
                return "%0.5f (+/-%0.03f)"% (mean, std * 2),params
            
            print()
            
        elif classifier=='nb':
            best_parameters=[ 
                        {'fit_prior': [clf.best_params_['fit_prior']], 'alpha': [clf.best_params_['alpha']]}
                       ]
            del clf
            clf = GridSearchCV(MultinomialNB(), best_parameters, cv=5,
                               scoring='%s' % score)
            clf.fit(X_train_a, y_train_a)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            print('----------')
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.5f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
                return "%0.5f (+/-%0.03f)"% (mean, std * 2),params
            print()

            
def global_word_score(metric,global_policy,word_fre_counter,topics):


    if global_policy=='sum':
        counter_global =Counter()
        for label in topics:
            counter_metric = metrics_update(metric,word_fre_counter[label][0],word_fre_counter[label][1])
    #         print(counter_metric[0])
            for k,v in counter_metric.items():
                counter_global[k]=counter_global[k]+v
    elif global_policy=='wsum':
        counter_global =Counter()
        for label in topics:
            counter_metric = metrics_update(metric,word_fre_counter[label][0],word_fre_counter[label][1])
            c1,c2 = sum(word_fre_counter[label][0].values()),sum(word_fre_counter[label][1].values())
            weight = np.float64(c1/(c1+c2))
    #         print(counter_metric[0],sum(word_fre_counter[label][0].values()),sum(word_fre_counter[label][1].values()),weight)
            for k,v in counter_metric.items():
                counter_global[k]=counter_global[k]+weight*v
    elif global_policy =='max':
        counter_global =Counter()
        for label in topics:
            counter_metric = metrics_update(metric,word_fre_counter[label][0],word_fre_counter[label][1])
#             print(counter_metric[0])
            for k,v in counter_metric.items():
                counter_global[k]=max(counter_global[k],v)
                
    return counter_global


def global_doc_score(metric,global_policy,doc_fre_counter,topics):


    if global_policy=='sum':
        counter_global =Counter()
        for label in topics:
            counter_metric = metrics_doc(metric,doc_fre_counter[label][0],doc_fre_counter[label][1])
    #         print(counter_metric[0])
            for k,v in counter_metric.items():
                counter_global[k]=counter_global[k]+v
    elif global_policy=='wsum':
        counter_global =Counter()
        for label in topics:
            counter_metric = metrics_doc(metric,doc_fre_counter[label][0],doc_fre_counter[label][1])
            c1,c2 = sum(word_fre_counter[label][0].values()),sum(word_fre_counter[label][1].values())
            weight = np.float64(c1/(c1+c2))
    #         print(counter_metric[0],sum(word_fre_counter[label][0].values()),sum(word_fre_counter[label][1].values()),weight)
            for k,v in counter_metric.items():
                counter_global[k]=counter_global[k]+weight*v
    elif global_policy =='max':
        counter_global =Counter()
        for label in topics:
            counter_metric = metrics_doc(metric,doc_fre_counter[label][0],doc_fre_counter[label][1])
#             print(counter_metric[0])
            for k,v in counter_metric.items():
                counter_global[k]=max(counter_global[k],v)
                
    return counter_global


def topic_features_selections(X,global_policy,num_of_topics,metrics,word_fre_counter,doc_fre_counter,Y,topics,model_num):
    
    if metrics in ['jsd','chi','mi','gr','ig']:
        counter_metric = global_word_score(metrics,global_policy,word_fre_counter,topics)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        labels = [i[0] for i in sorted_dict[-num_of_topics:]]
        print(labels)
    elif metrics in ['ext_jsd']:
        counter_metric = ext_jsd(word_fre_counter,X.shape[0],model_num)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        labels = [i[0] for i in sorted_dict[-num_of_topics:]]
        print(labels)
    else:
        counter_metric = global_doc_score(metrics,global_policy,doc_fre_counter,topics)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        labels = [i[0] for i in sorted_dict[-num_of_topics:]]
        print(labels)
        
    X_ = X[:,labels]
    X_norm = preprocessing.normalize(X_, norm='l1')
    
    score_svm_micro,_ = grid_search('svm',2018,0,['f1_micro'],X_norm,Y)
    score_svm_macro,_ = grid_search('svm',2018,0,['f1_macro'],X_norm,Y)
#     score_nb,_ = grid_search('nb',2018,0,measure,X_norm,Y)

#     return score_svm,score_nb
    return score_svm_micro,score_svm_macro

def random_topic_selections(num_of_topics,num,X,Y):
    topics = list(range(num_of_topics))
    labels = np.random.choice(topics,size=num,replace=False)
    
    print(labels)

    X_ = X[:,labels]
  
    X_norm = preprocessing.normalize(X_, norm='l1')

    score_svm_micro,_ = grid_search('svm',2018,0,['f1_micro'],X_norm,Y)
    score_svm_macro,_ = grid_search('svm',2018,0,['f1_macro'],X_norm,Y)
#     score_nb,params_nb = grid_search('nb',2018,0,['f1'],X_norm,Y)

    return float(score_svm_micro[:7]),float(score_svm_macro[:7])