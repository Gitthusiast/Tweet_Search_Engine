from ranker import Ranker
from math import log


# DO NOT MODIFY CLASS NAME
class Searcher:
    # Constants for accessing data lists
    DF_INDEX = 0

    DOCUMENT_ID_INDEX = 0
    FREQUENCY_INDEX = 1
    LENGTH_INDEX = 4  # document length

    # Constants for BM25+ calculation
    K1 = 1.2
    B = 0.75
    DELTA = 1

    TOP_N = 3

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None,
                 stemming=False, spell_correction=False, thesaurus=False, wordnet=False):

        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self.corpus_size = self._indexer.number_of_documents
        self.average_length = self._indexer.average_document_length

        # dynamically choose search engine implementations
        self.stemming = stemming
        self.spell_correction = spell_correction
        self.thesaurus = thesaurus
        self.wordnet = wordnet

    def calculate_doc_scores(self, term, relevant_docs):

        """
        Retrieves term's posting file and calculates score for each relevant document.
        Adds the relevant documents to relevant_docs dictionary
        :param term: query term for retrieval
        :param relevant_docs: dictionary of relevant documents
        :param posting_pointer: pointer (name) of relevant posting file
        :param posting_file: relevant posting file
        :return: returns a tuple of the current relevant posting pointer and posting file
        """

        inverted_document_frequency = log(self.corpus_size / self._indexer.inverted_idx[term][Searcher.DF_INDEX])  # idf

        if term in self._indexer.postingDict:
            documents = self._indexer.postingDict[term]
            for document in documents:

                # calculate score
                document_id = document[Searcher.DOCUMENT_ID_INDEX]
                doc_weight = document[Searcher.FREQUENCY_INDEX]
                normalized_length = document[Searcher.LENGTH_INDEX] / self.average_length

                if document_id not in relevant_docs:
                    relevant_docs[document_id] = 0

                # calculate score according to BM25+ weighting formula
                relevant_docs[document_id] += inverted_document_frequency * (
                        float((doc_weight * (Searcher.K1 + 1))) / (
                            doc_weight + Searcher.K1 *
                            (1 - Searcher.B + Searcher.B * normalized_length)) + Searcher.DELTA)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """

        # parse query according to the same parsing rules of the corpus
        entities = {}
        term_dict = {}
        parsed_query = self._parser.parse_sentence(query, entities, stemming=self.stemming)
        self._parser.parse_capital_letters(parsed_query, term_dict)
        processed_query = [*term_dict.keys()] + [*entities.keys()]

        # perform spell correction
        if self.spell_correction:

            from spellchecker import SpellChecker
            spell_checker = SpellChecker()
            corrected_terms = []

            # list all misspelled terms in the query
            misspelled_terms = spell_checker.unknown([*term_dict.keys()])
            for term in misspelled_terms:

                # only correct terms that aren't in the inverted dictionary
                # terms in the dictionary are considered correct for retrieval
                if term not in self._indexer.inverted_idx:
                    candidates = spell_checker.candidates(term)
                    max_to_return = min(Searcher.TOP_N, len(candidates))
                    candidates = candidates[:max_to_return]  # return only the top 3 results
                    if term in candidates:  # remove duplicate originally correct terms
                        candidates.remove(term)

                    for candidate in candidates:  # remove corrections already in query
                        if candidate in parsed_query:
                            candidates.remove(candidate)

                    corrected_terms.extend(candidates)

            processed_query += corrected_terms  # extend query with corrected words

        if self.thesaurus:

            from nltk.corpus import lin_thesaurus as thes

            candidates = []
            for term in processed_query:

                synsets = thes.synonyms(term)
                for synset in synsets:
                    synonyms = [*synset[1]]
                    if len(synonyms) > 0:
                        max_to_return = min(Searcher.TOP_N, len(synonyms))
                        best_synonyms = synonyms[:max_to_return]
                        for synonym in best_synonyms:
                            if synonym != term and synonym not in processed_query and synonym in self._indexer.inverted_idx:
                                candidates.append(synonym)  # extend the query
                        break

            processed_query += candidates

        if self.wordnet:

            from nltk.corpus import wordnet

            print("wordenting")
            candidates = []
            for term in processed_query:
                print(f"term {term}:")
                synsests = wordnet.synsets(term)  # retrieve best syn_sets
                max_to_return = min(Searcher.TOP_N, len(synsests))
                synsests = synsests[0:max_to_return]
                print("returned synsets")
                skip = False
                for synset in synsests:
                    for lemma in synset.lemmas()[:max_to_return]:  # possible synonyms
                        print(f"possible lemma: {lemma.name()}")
                        if lemma.name() != term and lemma.name() not in processed_query and lemma.name():
                            if lemma.name() in self._indexer.inverted_idx:
                                candidates.append(lemma.name())
                                print(f"appended {lemma.name()}")
                                skip = True
                                break
                            elif lemma.name().lower() in self._indexer.inverted_idx:
                                candidates.append(lemma.name())
                                print(f"appended {lemma.name()}")
                                skip = True
                                break
                            elif lemma.name().upper() in self._indexer.inverted_idx:
                                candidates.append(lemma.name())
                                print(f"appended {lemma.name()}")
                                skip = True
                                break

                    if skip:
                        break

            parsed_query += candidates

        # dictionary for holding all relevant documents (at least one query term appeared in the document)
        # format: {document_id: score}
        relevant_docs = {}
        for term in processed_query:

            # check if term exists in inverted dictionary in either lower or upper form
            if term in self._indexer.inverted_idx:
                self.calculate_doc_scores(term, relevant_docs)
            elif term.islower() and term.upper() in self._indexer.inverted_idx:
                self.calculate_doc_scores(term.upper(), relevant_docs)
            elif term.isupper() and term.lower() in self._indexer.inverted_idx:
                self.calculate_doc_scores(term.lower(), relevant_docs)

        n_relevant = len(relevant_docs)
        ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs)

        return n_relevant, ranked_doc_ids
