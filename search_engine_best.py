import pandas as pd
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import multiprocessing


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:
    SENTINEL = "STOP"

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None, stemming=True, spell_correction=False, thesaurus=True, wordnet=False):
        self._config = config
        self._parser = Parse()
        self._indexer = Indexer(config)
        self._model = None

        # dynamically choose search engine implementations
        self.stemming = stemming
        self.spell_correction = spell_correction
        self.thesaurus = thesaurus
        self.wordnet = wordnet

        # multiprocess shared objects initialized with lock
        self.number_of_documents = multiprocessing.Value('i', 0)
        self.total_document_length = multiprocessing.Value('i', 0)

    def _parse_document(self, unparsed_queue, parsed_queue, stemming=False):

        for document_as_list in iter(unparsed_queue.get, SearchEngine.SENTINEL):
            parsed_document = self._parser.parse_doc(document_as_list, stemming)
            parsed_queue.put(parsed_document)
            with self.total_document_length.get_lock():
                self.total_document_length.value += parsed_document.doc_length

            with self.number_of_documents.get_lock():
                self.number_of_documents.value += 1
            unparsed_queue.task_done()

        # send terminating sentinel task
        parsed_queue.put(SearchEngine.SENTINEL)  # send terminating sentinel task
        unparsed_queue.task_done()  # calling for the last sentinel JoinableQueue.get

    def _index_document(self, parsed_queue, indexer_queue):

        indexer = indexer_queue.get()
        for document in iter(parsed_queue.get, SearchEngine.SENTINEL):
            indexer.add_new_doc(document)
            parsed_queue.task_done()

        indexer_queue.put(indexer)
        parsed_queue.task_done()  # calling for the last sentinel JoinableQueue.get

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, corpus_path, toSave=False, save_path=None):
        """
            Builds the retrieval model.
            Preprocess, parse and index corpus.
            Reads parquet file and passes it to the parser, then indexer.
            Input:
                corpus_path - path to corpus parquet file
            Output:
                No output, just modifies the internal _indexer object.
            """

        df = pd.read_parquet(corpus_path, engine="pyarrow")
        documents_list = df.values.tolist()

        # read, parse and index documents as a pipeline.
        # Since we are working with a small dataset and the inverted index may remain in the RAM we can build a pipeline
        # of sub-processes each responsible for a different part of the flow
        unparsed_queue = multiprocessing.JoinableQueue()
        parsed_queue = multiprocessing.JoinableQueue()
        indexer_queue = multiprocessing.Queue()

        # Iterate over every document in the file
        #  for parsing documents
        parsing_process = multiprocessing.Process(target=self._parse_document,
                                                  args=(unparsed_queue, parsed_queue, self.stemming))
        parsing_process.start()
        print("Started accepting documents for parsing")

        # for indexing parsed documents
        indexing_process = multiprocessing.Process(target=self._index_document,
                                                   args=(parsed_queue, indexer_queue))
        indexing_process.start()
        indexer_queue.put(self._indexer)
        print("Started accepting parsed documents for indexing")

        # start sending into pipeline
        for document in documents_list:
            unparsed_queue.put(document)

        # send terminating sentinel task
        unparsed_queue.put(SearchEngine.SENTINEL)

        # waiting for parsing and indexing to finish
        unparsed_queue.join()
        parsed_queue.join()
        self._indexer = indexer_queue.get()  # retrieve updated indexer

        # after indexing all non-entity terms in the corpus, index legal entities
        self._indexer.index_entities()
        print("finished indexing entities")

        # after indexing the whole corpus, verify validity of all terms
        self._indexer.verify_posting()
        print("finished verifying posting")

        # calculate average document length
        average_document_length = float(self.total_document_length.value) / self.number_of_documents.value
        self._indexer.set_number_of_documents(self.number_of_documents.value)
        self._indexer.set_total_document_length(self.total_document_length.value)
        self._indexer.set_average_document_length(average_document_length)

        if toSave:
            self._indexer.save_index(save_path)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, path):
        """
        Loads a pre-computed index (or indices) so for query retrieval.
        Input:
            :param path - path to file of pickled index.
        :return tuple of (inverted_idx, postingDict)
        """
        self._indexer.inverted_idx, self._indexer.postingDict = self._indexer.load_index(path)
        return self._indexer.inverted_idx, self._indexer.postingDict

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and 
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """

        searcher = Searcher(self._parser, self._indexer, model=self._model,
                            stemming=self.stemming, spell_correction=self.spell_correction,
                            thesaurus=self.thesaurus, wordnet=self.wordnet)

        return searcher.search(query)
