import utils


# DO NOT MODIFY CLASS NAME
class Indexer:

    # Constants for accessing data lists
    DF_INDEX = 0
    TOTAL_FREQUENCY_INDEX = 1
    TOTAL_FREQUENCY_ENTITY = 1

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):

        self.test = 0
        self.config = config

        self.inverted_idx = {}  # dictionary format {term: [document_frequency, total_frequency] }

        # dictionary holds posting files for current ongoing documents
        # dictionary format: { term:
        #                       [[docId, term_frequency, max_term_frequency, unique_term_number,
        #                       document_length, doc_date]] }
        self.postingDict = {}

        # entity_dict is a partial inverted_idx dictionary containing only entities
        # entity_dict is of the format {entity: document_frequency, total_frequency}
        self.entity_dict = {}

        # statistical data for weighting and calculating similarity
        self.number_of_documents = 0
        self.total_document_length = 0
        self.average_document_length = 0

    def set_number_of_documents(self, number_of_documents):
        self.number_of_documents = number_of_documents

    def set_total_document_length(self, total_document_length):
        self.total_document_length = total_document_length

    def set_average_document_length(self, average_document_length):
        self.average_document_length = average_document_length

    def collect_possible_entities(self, document_entities):

        """
        Collect all possible entities from the possible entities dictionary in a document and add them to entity_dict
        A possible entity is defined as every sequence of tokens starting in a capital letter
        An entity is defined as a possible entity that appears in at least 2 different documents
        :param document_entities: dictionary of all entities in the document
        """

        for entity, frequency in document_entities.items():

            if entity != "":
                if entity not in self.entity_dict:
                    self.entity_dict[entity] = [1, frequency]
                else:
                    self.entity_dict[entity][Indexer.DF_INDEX] += 1
                    self.entity_dict[entity][Indexer.TOTAL_FREQUENCY_ENTITY] += frequency

    def index_uniform_terms(self, document_dictionary):

        """
        Index terms according to capital letters rule.
        Ensures a uniform appearance of terms across all corpus
        If a term only appears in capital form - record as upper case. Else, record in lower case
        Documents are indexed in batches
        :param document_dictionary - per document uniform term dictionary
        """

        for term, frequency in document_dictionary.items():

            # Add term to inverted_idx dictionary
            # In the dictionary keep the term_frequency
            # term_frequency - how many times the term appeared in the document
            # key indicates if term is capital or lower case

            # Check if term form is a upper case
            if term.isupper():

                term_lower_form = term.lower()
                # check in which form the token appears in dictionary and update it accordingly
                if term not in self.inverted_idx and term_lower_form not in self.inverted_idx:
                    self.inverted_idx[term] = [1, frequency]
                elif term in self.inverted_idx:
                    self.inverted_idx[term][Indexer.DF_INDEX] += 1
                    self.inverted_idx[term][Indexer.TOTAL_FREQUENCY_INDEX] += frequency
                else:  # term appears in lower case in dictionary
                    self.inverted_idx[term_lower_form][Indexer.DF_INDEX] += 1
                    self.inverted_idx[term_lower_form][Indexer.TOTAL_FREQUENCY_INDEX] += frequency

            # If current term is lower case, number or punctuation - change key to lower case
            else:

                term_upper_form = term.upper()
                # check in which form the token appears in dictionary and update it accordingly
                if term_upper_form not in self.inverted_idx and term not in self.inverted_idx:
                    self.inverted_idx[term] = [1, frequency]
                elif term_upper_form in self.inverted_idx:  # replace term in dictionary from upper case to lower case
                    if term.islower() or term.isupper():  # term is neither a number nor a punctuation
                        self.inverted_idx[term] = [self.inverted_idx[term_upper_form][Indexer.DF_INDEX] + 1,
                                                   self.inverted_idx[term_upper_form][
                                                       Indexer.TOTAL_FREQUENCY_INDEX] + frequency]
                        self.inverted_idx.pop(term_upper_form, None)  # remove upper case form from the dictionary
                    else:  # term is number or punctuation
                        self.inverted_idx[term][Indexer.DF_INDEX] += 1
                        self.inverted_idx[term][Indexer.TOTAL_FREQUENCY_INDEX] += frequency
                else:  # term appears in lower case in dictionary
                    self.inverted_idx[term][Indexer.TOTAL_FREQUENCY_INDEX] += frequency

    def add_document_to_posting(self, doc_id, max_tf, unique_terms_number, document_length, document_date,
                                document_dictionary, document_entities):

        """
        Creates a posting file.
        This function doesn't promise integrity of lower/upper letter rule or entities rule.
        Integrity of these rules should be enforced by after all documents has been parsed.
        Posting dictionary format is { term: [[docId, term_frequency]] }
        :param doc_id: current document id to be added
        :param max_tf: max term frequency of a term in the document
        :param unique_terms_number: number of unique terms in the document
        :param document_length: number of terms in the document with repetition
        :param document_date: str - document date
        :param document_dictionary: document's uniform term dictionary
        :param document_entities: document's entities dictionary
        :param batch_index: current documents index to create partial posting file
        """

        # unpack dict_items object into a list before applying operator +
        for term, frequency in [*document_dictionary.items()] + [*document_entities.items()]:

            if term not in self.postingDict:
                self.postingDict[term] = [[doc_id, frequency, max_tf, unique_terms_number,
                                          document_length, document_date]]
            else:
                self.postingDict[term].append([doc_id, frequency, max_tf, unique_terms_number,
                                              document_length, document_date])

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        Index all non-entity terms and collect all possible entities. Finally creates a posting file.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :type Document
        """

        # index all terms and possible entities in the document
        document_dictionary = document.term_doc_dictionary
        document_entities_dictionary = document.entities

        self.index_uniform_terms(document_dictionary)
        self.collect_possible_entities(document_entities_dictionary)

        # create posting file for  document
        doc_id = document.tweet_id
        max_tf = document.max_tf
        unique_term_number = document.unique_term_number
        document_length = document.doc_length
        document_date = document.date
        document_dictionary = document.term_doc_dictionary
        document_entities_dictionary = document.entities

        self.add_document_to_posting(doc_id, max_tf, unique_term_number, document_length, document_date,
                                     document_dictionary, document_entities_dictionary)

    def index_entities(self):

        """
        Index all legal entities recorded in the indexer's entity dictionary after processing all the corpus
        """

        for entity, frequencies in self.entity_dict.items():

            document_frequency = frequencies[Indexer.DF_INDEX]
            total_frequency = frequencies[Indexer.TOTAL_FREQUENCY_ENTITY]

            # check if possible entity is a legal entity and and index it
            if entity not in self.inverted_idx and document_frequency >= 2:
                self.inverted_idx[entity] = [document_frequency, total_frequency]

    def verify_posting(self):

        """
        Verify legality of terms in postings according to lower/upper case rule and entities rule.
        """

        # sort each posting file entry by document id
        for term in list(self.postingDict.keys()):

            # verify legality of the term

            # if term is an entity and it doesn't appear in at least two different document
            # remove from posting file
            if term in self.entity_dict and self.entity_dict[term][Indexer.DF_INDEX] == 1:
                self.postingDict.pop(term, None)
                continue

            # if term is upper case and there has been a term in lower form in the corpus
            # change the term in the posting to lower form
            elif term.isupper() and term not in self.inverted_idx:
                if term.lower() in self.postingDict.keys():
                    self.postingDict[term.lower()] += self.postingDict.pop(term)
                else:
                    self.postingDict[term.lower()] = self.postingDict.pop(term)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, path):
        """
        Loads a pre-computed index (or indices) so for query retrieval.
        Input:
            :param path - path to file of pickled index.
        :return tuple of (inverted_idx, postingDict)
        """
        return utils.load_obj(path)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, path):
        """
        Saves a pre-computed index (or indices) for saving inverted index (dictionary and posting).
        Pickles a tuple of (inverted_idx, postingDict)
        Input:
              :param: path - path to location where pickled index file should be saved.
        """

        inverted_index = (self.inverted_idx, self.postingDict)
        utils.save_obj(inverted_index, path)

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []
