# topic-based_corpus_comparison

### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.14.3](http://www.numpy.org/)
* Required: [scikit-learn >= 0.19.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 0.23.0](https://pandas.pydata.org/)
* Required: [gensim >= 3.6.0](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 2.2.2](https://matplotlib.org/)
* Required: [bokeh >= 0.12.16](https://bokeh.pydata.org/en/latest/)

### Basic Usage

To perform topic-based corpus comparison, the input corpus of documents should consist of plain text files stored in csv format (one corpus per file), each row corresponding to one document in that corpus and the column of text should be named "text", the format can be refered to the csv file in the directory "data/" (please ignore the column "label", it is used for document classification experiments). We assume your text data is preprocesed. Our code does not include the preprocessing component so that you can customize your own preprocessing strategy.

##### Step 1: Train LDA model

The first step of TBCC is train sevearl LDA models based on the data located in directory "data/" with different values of k(number of topics in model).  

	python build_lda_model.py 5 10 20 40 80 160

It should be noted that you must specify at least one k, or you can input a list of k (*5 10 15 20 25 30*) as shown in above command. There are some other opitons you can specify such as output directory (default is directory "LDA_MODELS/"), random seed (default is 1984) and so on, as shown below
	
	python build_lda_model.py --random_state=2020 5 10 20 40 80 160
	
For more details, you can type command 	
	
	python build_lda_model.py -h

##### Step 2: Train Fasttext word embedding model

After generating LDA models, the next step of TBCC is to build up a word embedding model for LDA model selection procedure. 

	python build_fasttext_model.py
	
The word embedding model will be stored in directory "FASTTEXT_MODEL/" by default, you can also play with other parameters. By command 
*python build_fasttext_model.py -h* you can see other options such as window size, dimensions.

##### Step 3: Analysis of topic coherence of topic models

The third step is caculating the topic coherence of each topic model for selecting the model with highest topic coherece.
	
	python compute_semantic_coherence.py 5 10 15 20 25 30
	
The result will be stored in the directory "SEMANTIC_COH/". It should be noted the if you change the random seed when training LDA models (default random seed is 1984), you need to specify it explicitly, since the LDA models are named after random seed and topic numbers.

	python compute_semantic_coherence.py --random-state=2020 5 10 15 20 25 30

##### Step 4: Conduct topic-based corpus comparison

After choosing the best k, we can conduct a topic-based corpus comparison according to various statistical discrimination metrics.

	python topic_based_cc.py -k 160 -s 1984 -m jsd
	
It should be noted the parameters *k, s, m* is mentory here indicating the number of topics, random_state (these two for targeting the topic model) and the employed statistical discrimination metrics (options are jsd, ext_jsd, chi, rf, ig, gr). The output will be stored in directory "COMPARISON_RESULT/" as well as shown in the console:


![alt text](https://github.com/GeorgeLuImmortal/topic-based_corpus_comparison/blob/master/COMPARISON_RESULT/comparison_result.png)


The first column is the index of topic, the second colmun is the words for charactering the topic, and the third colmun is the corpus index which the topic belongs to.

##### Step 5: Visualization

We can also visualizae the result via scatter plot exploiting t-SNE to project documents into a 2-D plane. The parameters are the same as above for example, k is the number of topic, s is the random state and m means the metric applied.

	python visualization.py -k 200 -s 1984 -m jsd

![alt text](https://github.com/GeorgeLuImmortal/topic-based_corpus_comparison/blob/master/VISUALIZATION/scatter_plot.png)

It will generate a html file under the directory "VISUALIZATION/" by default, you can open the file using any browser. The different color indicates documents from different corpora and the number is the index of the most discriminative topics selected by TBCC. You can refer to the file in directory "COMPARISON_RESULT/" for the contents of topics, or you can use mouth hover through the html page for the discriptors of the topics as shown in the above figure (this is a result of Chi-square, 200 topics).
