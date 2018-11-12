import itertools
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import FastText
import pandas as pd
import numpy as np
import gensim
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser




def calculate_semantic_coherence(lda_model, word_emb_model,no_of_term,no_of_topic):
    global_sim = 0.0
    for topic_index in range(no_of_topic):
        descriptors_ids = [i[0] for i in lda_model.show_topic(topic_index,no_of_term)]
        
        sim = 0.0
        num_of_pairs = 0
        for pair in itertools.combinations(descriptors_ids,2):
            word1 = pair[0]
            word2 = pair[1]
            try:
                sim+=word_emb_model.similarity(word1, word2)
            except Exception:
                sim+=0
            num_of_pairs+=1
        global_sim+=(sim/num_of_pairs)
    
    return global_sim/no_of_topic




def compute_semantic_draw(lda_dir,loaded_model,nums,output_dir,seed,no_of_term):
# corpus_set = ['wikilow','bbc','bbcsport','guardian2013','irishtimes2013','nytimes1999','nytimes2013']
# nums = [5,10,20,50,70,100,150,200,250,300,350,400,450,500,550,600]
    
    coherence = []
    for num in nums:
        try:
            print('compute coherence for %d'%(num))
            lda_model = gensim.models.ldamodel.LdaModel.load(lda_dir+'seed_%d_topic_%d'%(seed,num))
            coherence.append(calculate_semantic_coherence(lda_model,loaded_model,no_of_term,num))
        except Exception:
            print("lda model doesn't exist")
            exit()

    t = [num for num in nums]
    s = coherence

    fig, ax = plt.subplots(figsize=(20,15))

    ax.plot(t, s)
    ax.set(xlabel='num of topics', ylabel='semantic coherence',
           title='Semantic Coherence of model')
    ax.grid()

    fig.savefig(output_dir+"semantic_coh.png")
    plt.show()

    df_coh = pd.DataFrame()
    df_coh['num'] = nums
    df_coh['coh'] = coherence

    df_coh.to_csv(output_dir+"semantic_coh.csv")


def main():

    parser = OptionParser(usage="usage: %prog [options] k1 k2 ...")
    parser.add_option("-f","--inputdir_ftmodel", action="store", type="string", dest="dirftmodel_input", help="Input directory for Fasttext model (default is FASTTEXT_MODEL/ directory)", default='FASTTEXT_MODEL/')
    parser.add_option("-l","--inputdir_ldamodel", action="store", type="string", dest="dirLDAmodel_input", help="Input directory for lda model (default is LDA_MODELS/ directory)", default='LDA_MODELS/')
    parser.add_option("-i","--random_state", action="store", type="int", dest="random_state", help="The random seed for training lda model (default is 1984)", default=1984)
    parser.add_option("-n","--n_terms", action="store", type="int", dest="no_of_term", help="The number of terms for computing semantic coherence (default is 10)", default=10)
    parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="Output directory (default is SEMANTIC_COH/ directory)", default='SEMANTIC_COH/')
   
    (options, args) = parser.parse_args()

    if( len(args) < 1 ):
        parser.error( "Must specify at least one number of topics" )   
    log.basicConfig(level=20, format='%(message)s')

    print(args)
    print('-'*6+'begin loading fasttext model'+'-'*6)
    loaded_model = FastText.load(options.dirftmodel_input+'ftmodel')
    print(loaded_model)

    nums = [int(num) for num in args]
    print('-'*6+'begin compute topic semantic coherence'+'-'*6)
    compute_semantic_draw(options.dirLDAmodel_input,loaded_model,nums,options.dir_out,options.random_state,options.no_of_term)

            
# --------------------------------------------------------------

if __name__ == "__main__":
    main()

