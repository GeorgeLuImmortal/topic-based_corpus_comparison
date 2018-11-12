# topic-based_corpus_comparison

### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.14.3](http://www.numpy.org/)
* Required: [scikit-learn >= 0.19.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 0.23.0](https://pandas.pydata.org/)
* Required: [gensim >= 3.6.0](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 2.2.2](https://matplotlib.org/)

### Basic Usage

To perform topic-based corpus comparison, the input corpus of documents should consist of plain text files stored in csv format (one corpus per file), each row corresponding to one document in that corpus and the column of text should be named "text", the format can be refered to the csv file in the directory "data/" (please ignore the column "label", it is used for document classification experiments). We assume your text data is preprocesed. Our code does not include the preprocessing component so that you can customize your own preprocessing strategy.

##### Step 1: Train LDA model

The first step of TBCC is train sevearl LDA models based on the data located in directory "data/" with different values of k(number of topics in model).  

	python build_lda_model.py 5 10 15 20 25 30

It should be noted that you must specify at least one k, or you can input a list of k as shown in above command. There are some other opitons you can specify such as input, output directory (default is directory "LDA_MODELS"), random seed and so on.
	
	python build_lda_model.py --random_state=2020 5 10 15 20 25 30
	
For more details, you can type command 	
	
	python build_lda_model.py -h

##### Step 2: Train Fasttext word embedding model

After generating LDA models, the next step of TBCC is to build up a word embedding model for LDA model selection procedures.



