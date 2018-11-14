import itertools
import matplotlib.pyplot as plt
from gensim.models import FastText
import pandas as pd
import numpy as np
import gensim
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser
import pandas as pd
import re
from collections import Counter
from sklearn import preprocessing
import topic_cc_utilities as util
import operator
import csv

def main():

    parser = OptionParser()
    parser.add_option("-i","--inputdir", action="store", type="string", dest="dir_input", help="Input directory for text (default is data/ directory)", default='data/')
    parser.add_option("-l","--inputdir_ldamodel", action="store", type="string", dest="dirLDAmodel_input", help="Input directory for lda model (default is LDA_MODELS/ directory)", default='LDA_MODELS/')
    parser.add_option("-k","--no_of_topics", action="store", type="int", dest="no_of_topics", help="The number of topics in lda model")
    parser.add_option("-s","--random_seed", action="store", type="int", dest="random_seed", help="The random seed in lda model ")
    parser.add_option("-m","--metric", action="store", type="string", dest="metric", help="The discimination metric, options are jsd, ext_jsd, chi, rf, ig, gr")
    parser.add_option("-g","--global_policy", action="store", type="string", dest="global_policy", help="The global policy : sum, wsum, max (default is max)",default='max')
    parser.add_option("-n","--no_of_d_topics", action="store", type="int", dest="no_of_d_topics", help="The number of discrimination topics shown  (default is 10)",default=10)
    parser.add_option("-d","--no_of_descriptors", action="store", type="int", dest="no_of_descriptors", help="The number of topic descriptors shown (default is 10)",default=10)
    parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="Output directory (default is COMPARISON_RESULT/ directory)", default='COMPARISON_RESULT/')
    
   
   
    # Parse command line arguments
    (options, args) = parser.parse_args()

    

    if options.no_of_topics is None:
        parser.error( "Must specify the number of topic" )   
    log.basicConfig(level=20, format='%(message)s')

    if options.random_seed is None:
        parser.error( "Must specify the random seed" )   
    log.basicConfig(level=20, format='%(message)s')

    if options.metric not in ['jsd', 'ext_jsd', 'chi', 'rf', 'ig', 'gr']:
        parser.error( "Must specify one discrimination metric from one of them: jsd, ext_jsd, chi, rf, ig, gr" )   
    log.basicConfig(level=20, format='%(message)s')
    
    path = options.dir_input
    corpora=[]
    labels=[]
    for idx,filename in enumerate(os.listdir(path)):
        df_corpus = pd.read_csv(path+filename)
        corpora = corpora + df_corpus.text.tolist()
        labels = labels + [idx]*len(df_corpus)

    corpora = [d.split(' ') for d in corpora]

    df_corpora = pd.DataFrame()
    df_corpora['text'] = corpora
    df_corpora['label'] = labels

    # Create Dictionary
    word_ids = gensim.corpora.Dictionary(corpora)

    # Term Document Frequency
    bow = [word_ids.doc2bow(doc) for doc in (corpora)]

    lda_model = gensim.models.ldamodel.LdaModel.load(options.dirLDAmodel_input+'seed_%d_topic_%d'%(options.random_seed,options.no_of_topics))

    
    ## topic distirbution for all metrics except rf
    all_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
    all_topics_csr = gensim.matutils.corpus2csc(all_topics)
    all_topics_numpy = all_topics_csr.T.toarray()

    X = np.float64(preprocessing.normalize(all_topics_numpy, norm='l1', axis=1))
    Y = labels

    
    word_fre_counter = util.one_versus_all(list(set(labels)),X,df_corpora)



    ## topic distirbution for rf

    threshold = 1e-2
    all_topics = lda_model.get_document_topics(bow, minimum_probability=threshold)
    all_topics_csr = gensim.matutils.corpus2csc(all_topics)
    all_topics_numpy = all_topics_csr.T.toarray()



    X_doc = np.float64(preprocessing.normalize(all_topics_numpy, norm='l1', axis=1))
   
    doc_fre_counter = util.one_versus_all_doc(list(set(labels)),X_doc,df_corpora)


    metrics = options.metric
    global_policy = options.global_policy
    sections = list(set(labels))
    model_num = options.no_of_topics
    if metrics in ['jsd','chi','gr','ig']:
        counter_metric = util.global_word_score(metrics,global_policy,word_fre_counter,sections)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        distinct_topics = [i[0] for i in sorted_dict[::-1]]
        
        disctinct_descriptors=[]
        sources = []       
        for i in distinct_topics[:options.no_of_d_topics]:
            descriptors = [tuple_[0] for tuple_ in lda_model.show_topic(i,options.no_of_descriptors)]
            disctinct_descriptors.append(descriptors)
            source_index = np.argmax([word_fre_counter[source][0][i]/sum(word_fre_counter[source][0].values()) for source in word_fre_counter])
            sources.append(source_index)
            print(i,'-'*6,descriptors,'-'*6,source_index)

        df_to_csv = pd.DataFrame()
        df_to_csv['topic'] = [i for i in distinct_topics[:options.no_of_d_topics]]
        df_to_csv['descriptors'] = disctinct_descriptors
        df_to_csv['sources'] = sources
        df_to_csv.to_csv(options.dir_out+'%s_distinct_topics.csv'%options.metric,index=False)
                         
    elif metrics in ['ext_jsd']:
        counter_metric = util.ext_jsd(word_fre_counter,X.shape[0],model_num)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        distinct_topics = [i[0] for i in sorted_dict[::-1]]
       
        disctinct_descriptors=[]
        sources = []       
        for i in distinct_topics[:options.no_of_d_topics]:
            descriptors = [tuple_[0] for tuple_ in lda_model.show_topic(i,options.no_of_descriptors)]
            disctinct_descriptors.append(descriptors)
            source_index = np.argmax([word_fre_counter[source][0][i]/sum(word_fre_counter[source][0].values()) for source in word_fre_counter])
            sources.append(source_index)
            print(i,'-'*6,descriptors,'-'*6,source_index)

        df_to_csv = pd.DataFrame()
        df_to_csv['topic'] = [i for i in distinct_topics[:options.no_of_d_topics]]
        df_to_csv['descriptors'] = disctinct_descriptors
        df_to_csv['sources'] = sources
        df_to_csv.to_csv(options.dir_out+'%s_distinct_topics.csv'%options.metric,index=False)
    else:
        counter_metric =util.global_doc_score(metrics,global_policy,doc_fre_counter,sections)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        distinct_topics = [i[0] for i in sorted_dict[::-1]]

        disctinct_descriptors=[]
        sources = []       
        for i in distinct_topics[:options.no_of_d_topics]:
            descriptors = [tuple_[0] for tuple_ in lda_model.show_topic(i,options.no_of_descriptors)]
            disctinct_descriptors.append(descriptors)
            source_index = np.argmax([word_fre_counter[source][0][i]/sum(word_fre_counter[source][0].values()) for source in word_fre_counter])
            sources.append(source_index)
            print(i,'-'*6,descriptors,'-'*6,source_index)

        df_to_csv = pd.DataFrame()
        df_to_csv['topic'] = [i for i in distinct_topics[:options.no_of_d_topics]]
        df_to_csv['descriptors'] = disctinct_descriptors
        df_to_csv['sources'] = sources
        df_to_csv.to_csv(options.dir_out+'%s_distinct_topics.csv'%options.metric,index=False)
# --------------------------------------------------------------

if __name__ == "__main__":
    main()















