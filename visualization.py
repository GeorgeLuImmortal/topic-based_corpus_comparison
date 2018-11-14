import numpy as np
from sklearn import preprocessing
import topic_cc_utilities as util
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import FastText
import pandas as pd
import numpy as np
import gensim
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser
import operator
import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.models import HoverTool
from bokeh import palettes
from bokeh.palettes import Dark2_5 as palette_dark25
import itertools


def visualization(tsne_lda,_lda_keys,descriptors,divergent_keys,divergent_topics,labels,outdir,model_num):
    
    df = pd.DataFrame(
    {
        "descriptors": descriptors,
        "topic": [str(i) for i in divergent_keys],
        "x": tsne_lda[:,0],
        "y": tsne_lda[:,1],
        
    }
    )
    
    df_cor = pd.DataFrame()
    df_cor['x']=tsne_lda[:,0]
    df_cor['y']=tsne_lda[:,1]
    df_cor['topics']=_lda_keys
    
    coordinates=[]
    for topic in divergent_topics:
        temp_df = df_cor[df_cor['topics']==topic]
        x_cor = np.average( temp_df['x'].tolist())
        y_cor = np.average( temp_df['y'].tolist())
        coordinates.append([x_cor,y_cor])
        
        
    source = bpl.ColumnDataSource(df)
    
    unique_labels = list(set(labels))
    
    colors = itertools.cycle(palette_dark25)  
    
    palette=[next(colors) for i in range(len(unique_labels))]
    palette= palette+['gray']
    
    color_map = bmo.CategoricalColorMapper(factors=[str(i) for i in unique_labels]+['others'],
                                       palette=palette)

    TOOLS ="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"


    p = bpl.figure(plot_width=2000, plot_height=2000,
    #                      title=title,
                         tools=TOOLS,
                         x_axis_type=None, y_axis_type=None, min_border=1)

    p.scatter(x='x', y='y',
              color={'field': 'topic', 'transform': color_map},
              legend='topic', source=source,fill_alpha=0.3,radius=0.2)
    
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = """<font size='5pt'>Topic: @topic <br /> Descriptors: @descriptors</font>"""
    
    for i in range(len(divergent_topics)):
        p.text(coordinates[i][0]-5, coordinates[i][1], [divergent_topics[i]],text_color="black",text_font_size='50pt',text_font_style='bold')
    
    bpl.output_file(outdir+"%s_%d_topics.html"%('visualization',model_num))
    bpl.show(p)

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
    parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="Output directory (default is VISUALIZATION/ directory)", default='VISUALIZATION/')
    
   
   
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
    random_seed=options.random_seed
    no_of_topics = options.no_of_topics
    metrics = options.metric
    global_policy = options.global_policy
    no_of_d_topics=options.no_of_d_topics
    model_num = no_of_topics

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

    lda_model = gensim.models.ldamodel.LdaModel.load(options.dirLDAmodel_input+'seed_%d_topic_%d'%(random_seed,no_of_topics))


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

    X_short = np.where(X>0.05,X,0)

    print('-'*12, "begin computing TSNE",'-'*12)

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

    # 20-D -> 2-D
    tsne_lda = tsne_model.fit_transform(X_short)


    sections = list(set(labels))
    if metrics in ['jsd','chi','gr','ig']:
        counter_metric = util.global_word_score(metrics,global_policy,word_fre_counter,sections)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        distinct_topics = [i[0] for i in sorted_dict[::-1]]

        divergent_topics = distinct_topics[:no_of_d_topics]
        _lda_keys = []
        divergent_keys = []
        for i in range(X.shape[0]):
            _lda_keys +=  X[i].argmax(),
            if X[i].argmax() in divergent_topics:
                divergent_keys+=  labels[i],
            else:
                divergent_keys += 'others',

        topic_summaries = []

        for i in range(X.shape[1]):  

            temp = [word[0] for word in lda_model.show_topic(i,8)]
            temp_1 = ' '.join(temp[:4])
            temp_2 = ' '.join(temp[4:])
            final = temp_1+'\n'+temp_2
            topic_summaries.append(final)

        descriptors=[topic_summaries[i] for i in _lda_keys]

        

    elif metrics in ['ext_jsd']:
        counter_metric = util.ext_jsd(word_fre_counter,X.shape[0],model_num)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        distinct_topics = [i[0] for i in sorted_dict[::-1]]

        divergent_topics = distinct_topics[:no_of_d_topics]
        _lda_keys = []
        divergent_keys = []
        for i in range(X.shape[0]):
            _lda_keys +=  X[i].argmax(),
            if X[i].argmax() in divergent_topics:
                divergent_keys+=  labels[i],
            else:
                divergent_keys += 'others',

        topic_summaries = []

        for i in range(X.shape[1]):  

            temp = [word[0] for word in lda_model.show_topic(i,8)]
            temp_1 = ' '.join(temp[:4])
            temp_2 = ' '.join(temp[4:])
            final = temp_1+'\n'+temp_2
            topic_summaries.append(final)

        descriptors=[topic_summaries[i] for i in _lda_keys]


    else:
        counter_metric =util.global_doc_score(metrics,global_policy,doc_fre_counter,sections)
        sorted_dict = sorted(counter_metric.items(), key=operator.itemgetter(1))
        distinct_topics = [i[0] for i in sorted_dict[::-1]]

        divergent_topics = distinct_topics[:no_of_d_topics]
        _lda_keys = []
        divergent_keys = []
        for i in range(X.shape[0]):
            _lda_keys +=  X[i].argmax(),
            if X[i].argmax() in divergent_topics:
                divergent_keys+=  labels[i],
            else:
                divergent_keys += 'others',

        topic_summaries = []

        for i in range(X.shape[1]):  

            temp = [word[0] for word in lda_model.show_topic(i,8)]
            temp_1 = ' '.join(temp[:4])
            temp_2 = ' '.join(temp[4:])
            final = temp_1+'\n'+temp_2
            topic_summaries.append(final)

        descriptors=[topic_summaries[i] for i in _lda_keys]
    
    print('-'*12, "begin visualization",'-'*12)
    visualization(tsne_lda,_lda_keys,descriptors,divergent_keys,divergent_topics,labels,options.dir_out,model_num)

       
# --------------------------------------------------------------

if __name__ == "__main__":
    main()




