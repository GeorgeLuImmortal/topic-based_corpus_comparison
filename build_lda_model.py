from gensim.models import FastText
import pandas as pd
import numpy as np
import gensim
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser


# --------------------------------------------------------------

def main():

    parser = OptionParser(usage="usage: %prog [options] k1 k2 ...")
    parser.add_option("-i","--inputdir", action="store", type="string", dest="dir_input", help="Input directory (default is data/ directory)", default='data/')
    parser.add_option("-s","--random_state", action="store", type="int", dest="random_state", help="The random seed for training lda model (default is 1984)", default=1984)
    parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="Output directory (default is LDA_MODELS/ directory)", default='LDA_MODELS/')
   
    # Parse command line arguments
    (options, args) = parser.parse_args()

    if( len(args) < 1 ):
        parser.error( "Must specify at least one number of topics" )   
    log.basicConfig(level=20, format='%(message)s')
    
    print('input dir is '+options.dir_input)
    print('output dir is '+options.dir_out)
    print(args)

    path = options.dir_input


    print('-'*6+'begin loading data'+'-'*6)

    corpora=[]
    for filename in os.listdir(path):
        df_corpus = pd.read_csv(path+filename)
        corpora = corpora + df_corpus.text.tolist()

    print('-'*6+'%s corpora are detected'%(str(len(os.listdir(path))))+'-'*6)
    print('-'*6+'%s documents in total'%(str(len(corpora)))+'-'*6)


    print('-'*6+'begin training model'+'-'*6)
    corpora = [d.split(' ') for d in corpora]
   
    # Create Dictionary
    word_ids = gensim.corpora.Dictionary(corpora)

    # Term Document Frequency
    bow = [word_ids.doc2bow(doc) for doc in (corpora)]

    nums = [int(num) for num in args]
    seed=options.random_state
 
    for num in nums:
        print('-'*6+'num of topics is %d'%num+'-'*6)
        print('-'*6+'random state is %d'%seed+'-'*6)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=bow,
                                                       id2word=word_ids,
                                                       num_topics=num, 
                                                       random_state=seed,
                                                       update_every=10,
                                                       chunksize=100,
                                                       passes=10,
                                                       alpha='auto',
                                                       per_word_topics=True)


        lda_model.save(options.dir_out+'seed_%d_topic_%d'%(seed,num))

            
# --------------------------------------------------------------

if __name__ == "__main__":
    main()








