# topic-based_corpus_comparison

### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.14.3](http://www.numpy.org/)
* Required: [scikit-learn >= 0.19.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 0.23.0](https://pandas.pydata.org/)
* Required: [gensim >= 3.6.0](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 2.2.2](https://matplotlib.org/)

### Basic Usage

To perform topic-based corpus comparison, the input corpus of documents should consist of plain text files stored in csv format (one corpus per file), each row corresponding to one document in that corpus and the column of text should be named "text"

The dynamic topic modeling process consists of three steps, discussed below. The archive 'data/sample.zip' contains a sample corpus of 1,324 news articles divided into three time windows (month1, month2, month3), which is used to illustrate these steps.

##### Step 1: Pre-processing
Before applying dynamic topic modeling, the first step is to pre-process the documents from each time window (i.e. sub-directory), to produce a *document-term matrix* for those windows. This involves tokenizing the documents, removing common stop-words, and building a document-term matrix for the time window. In the example below, we parse all .txt files in the sub-directories of 'data/sample'. The output files will be stored in the directory 'data'. Note that the final options below indicate that we want to apply TF-IDF term weighting and document length normalization to the documents before writing each matrix.

	python prep-text.py data/sample/month1 data/sample/month2 data/sample/month3 -o data --tfidf --norm

The result of this process will be a collection of Joblib binary files (*.pkl and *.npy) written to the directory 'data', where the prefix of each corresponds to the name of each time window (e.g. month1, month2 etc).

##### Step 2: Window Topic Modeling 
Once the data has been pre-processed, the next step is to generate the *window topics*, where a topic model is created by applying NMF to each the pre-process data for each time window. For the example data, we apply it to the three months. If we want to use the same number of topics for each window (e.g. 5 topics), we can run the following, where results are written to the directory 'out':
	
	python find-window-topics.py data/*.pkl -k 5 -o out

When the process has completed, we can view the descriptiors (i.e. the top ranked terms) for the resulting window topics as follows:

	python display-topics.py out/month1_windowtopics_k05.pkl out/month2_windowtopics_k05.pkl out/month3_windowtopics_k05.pkl

The top terms and document IDs can be exported from a NMF results file to two individual comma-separated files using 'export-csv.py'. For instance, to export the top 50 terms and document IDs for a single results file:

	python export-csv.py out/month1_windowtopics_k05.pkl -t 50

##### Step 3: Dynamic Topic Modeling 
