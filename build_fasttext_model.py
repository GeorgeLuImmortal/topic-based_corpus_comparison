from gensim.models import FastText
import pandas as pd
import numpy as np
import gensim
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser


# --------------------------------------------------------------

def main():

	parser = OptionParser()
	parser.add_option("-i","--inputdir", action="store", type="string", dest="dir_input", help="Input directory for text (default is data/ directory)", default='data/')
	parser.add_option("-c","--mincount", action="store", type="int", dest="min_count", help="The model ignores all words with total frequency lower than this (default is 1)", default=1)
	parser.add_option("--workers", action="store", type="int", dest="workers", help="Number of workers used in training (default is -1 means using all threads)", default=-1)
	parser.add_option("-d","--dimensions", action="store", type="int", dest="dimensions", help="The dimensionality of the word vectors (default is 100)", default=100)
	parser.add_option("-w","--window", action="store", type="int", dest="w2v_window", help="The maximum distance for model to use between the current and predicted word within a sentence (default is 5)", default=5)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="Output directory (default is FASTTEXT_MODEL/ directory)", default='FASTTEXT_MODEL/')
	parser.add_option("-m", action="store", type="int", dest="model_type", help="Type of word embedding model to build (default is 1 means sg)", default=1)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	
	print('input dir is '+options.dir_input)
	print('output dir is '+options.dir_out)

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
	model = FastText(corpora, size=options.dimensions, window=options.w2v_window, min_count=options.min_count, workers=options.workers,iter=10,sg=options.model_type,negative=5,ns_exponent=0.75,sample=1e-5)
	model.save(options.dir_out+'ftmodel')
	
	print(model)

			
# --------------------------------------------------------------

if __name__ == "__main__":
	main()




