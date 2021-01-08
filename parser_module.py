from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from document import Document
from string import punctuation
from demoji import findall
import stemmer


class Parse:

    numeric_counter = 0

    months = {"jan": "1", "january": "1", "feb": "2", "february": "2", "mar": "3", "march": "3", "apr": "4",
              "april": "4", "may": "5", "jun": "6", "june": "6", "jul": "7", "july": "7", "aug": "8", "august": "8",
              "sep": "9", "september": "9", "oct": "10", "october": "10", "nov": "11", "november": "11", "dec": "12",
              "december": "12"}

    days = {"first": "1", "1st": "1", "second": "2", "2nd": "2", "third": "3", "fourth": "4", "4th": "4", "fifth": "5",
            "5th": "5", "sixth": "6", "6th": "6", "seventh": "7", "7th": "7", "eighth": "8", "8th": "8", "ninth": "9",
            "9th": "9", "tenth": "10", "10th": "10", "eleventh": "11", "11th": "11", "twelfth": "12", "12th": "12",
            "thirteenth": "13", "13th": "13", "fourteenth": "14", "14th": "14", "fifteenth": "15", "15th": "15",
            "sixteenth": "16", "16th": "16", "seventeenth": "17", "17th": "17", "eighteenth": "18", "18th": "18",
            "nineteenth": "19", "19th": "19", "twentieth": "20", "twenty": "20", "20th": "20", "twenty-first": "21",
            "21tst": "21", "22nd": "22", "twenty-second": "22", "23rd": "23", "twenty-third": "23", "24th": "24",
            "twenty-fourth": "24", "25th": "25", "twenty-fifth": "25", "26th": "26", "twenty-sixth": "26",
            "27th": "27", "twenty-seventh": "27", "28th": "28", "twenty-eighth": "28", "twenty-ninth": "29",
            "29th": "29", "30th": "30", "thirty": "30", "31st": "31", "thirty-first": "31"}

    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(["rt", "n't", "'re", "gon", "na", "covid", "coronavirus", "covid-19"])
        self.punctuation_to_remove = punctuation.replace('#', '').replace('@', '').replace('%', '').replace('$', '')
        self.symbols = "<>:\"/\\|!?*~.'`-_()^,+=;"
        self.token_stemmer = stemmer.Stemmer()

    def get_valid_url(self, url_col):
        """
        :param url_col: "urls" column or "retweet_urls" or "quote_urls" columns
        :return: pure valid url or empty string if no valid url was present. format - {"":"return_value"}
        """

        if url_col != "{}":
            trans_table = url_col.maketrans("\"", " ")
            urls = url_col.translate(trans_table)
            urls = urls.split()
            if len(urls) == 5:
                return urls[3]
        return ""

    def parse_hashtag_underscore(self, text_tokens, i):
        """
        this function deals with hashtags of the form #stay_at_home
        :param text_tokens: list of tokens that is changed according to the given rules
        :param i: the index of the "#" token
        """
        token = text_tokens[i + 1]
        del text_tokens[i + 1]
        joined_hashtag = '#'
        insertion_index = 0
        num_inserted = 0
        splited_tokens = token.split("_")
        for j in range(len(splited_tokens)):
            if splited_tokens[j] != "":
                text_tokens.insert(i + 1 + insertion_index, splited_tokens[j].lower())
                insertion_index += 1
                num_inserted += 1
                joined_hashtag += splited_tokens[j]
        text_tokens[i] = joined_hashtag

    def parse_hashtag_camel_case(self, text_tokens, i):
        """
        this function parses hashtags of the the type #StayAtHome #stayAtHome
        :param text_tokens: list of tokens that is changed
        :param i: "#" index
        """
        token = text_tokens[i + 1]
        del text_tokens[i + 1]
        j = 0
        joined_hashtag = '#'
        from_index = 0
        insertion_index = 0
        while j < len(token):
            if token[j].isupper() and j != 0:
                text_tokens.insert(i + 1 + insertion_index, token[from_index:j].lower())
                joined_hashtag += token[from_index:j].lower()
                from_index = j
                insertion_index += 1
            j += 1
        if token[from_index:len(token)] != '':
            joined_hashtag += token[from_index:len(token)].lower()
            text_tokens.insert(i + 1 + insertion_index, token[from_index:len(token)].lower())
        text_tokens[i] = joined_hashtag

    def parse_hashtag_upper_case(self, text_tokens, i):

        """
        this function parses hashtags of the the type #COVID19 #NJ
        :param text_tokens: list of tokens that is changed
        :param i: "#" index
        """
        joined_hashtag = '#'
        joined_hashtag += text_tokens[i+1].lower()
        text_tokens[i] = joined_hashtag  # "#covid19"

    def parse_hashtag(self, text_tokens, i):
        """
        this function calls to parse underscore or parse camel case respectively
        :param - i the index of
        :return - return False if the hashtag contained not ascii values else return True
        """

        if len(text_tokens) > i + 1 and not text_tokens[i+1].isascii():
            del text_tokens[i]  # deleting ashtag
            del text_tokens[i]  # deleting not ascii symbol
            return False

        # parsing snake case
        if len(text_tokens) > i + 1 and text_tokens[i + 1].count('_') > 0:
            self.parse_hashtag_underscore(text_tokens, i)

        elif len(text_tokens) > i+1 and text_tokens[i+1].isupper():
            self.parse_hashtag_upper_case(text_tokens, i)

        # parsing pascal and camel cases
        elif len(text_tokens) > i + 1:
            self.parse_hashtag_camel_case(text_tokens, i)
        return True

    def parse_tagging(self, text_tokens, i):
        """
        this function appends @ and name that our tokenizer separates
        :param text_tokens: list of tokens
        :param i: index of '@'
        :return:
        """
        if len(text_tokens) > i + 1:
            text_tokens[i] += text_tokens[i + 1]
            del text_tokens[i + 1]

    def parse_url(self, text_tokens, i):
        """
        this function parses url according to the rules.
        :param text_tokens: list of tokens
        :param i: index of "https"
        """
        del text_tokens[i]  # removing 'https or http'
        if len(text_tokens) > i and text_tokens[i] == ":":
            if text_tokens[i] == ':':
                del text_tokens[i]  # removing ':'

                link_token = text_tokens[i]

                tokens_in_url = link_token.split("/")
                del text_tokens[i]

                token_index = 0
                while token_index < len(tokens_in_url):
                    if tokens_in_url[token_index] == "t.co":
                        break
                    if tokens_in_url[token_index] != "twitter.com" and tokens_in_url[token_index] != "":
                        text_tokens.insert(i + token_index, tokens_in_url[token_index].lstrip("w."))
                    token_index += 1

    def is_float(self, number):

        """
        Verify if a string can be converted to float
        :param number - string to be converted
        :return Boolean - can be converted or not
        """

        try:
            float(number.replace(",", ""))
            if number.lower() != "infinity":
                return True
        except ValueError:
            return False

    def parse_numeric_values(self, text_tokens, index):

        """
        Parse numeric tokens according to specified rules.
        Any number in the thousands, millions and billions will be abbreviated to #K, #M and #B respectively
        Any number signifying percentage will be shown as #%
        Fractions of the format #/# will stay the same
        :param text_tokens: list of tokens to be parsed
        :param index: index of currently parsed token
        """

        self.numeric_counter += 1
        token = text_tokens[index]
        numeric_token = float(token.replace(",", ""))

        # format large numbers
        # any number in the thousands, millions and billions will be abbreviated to #K, #M and #B respectively
        if 1000 <= numeric_token < 1000000:
            formatted_token = "{num:.3f}".format(num=(numeric_token / 1000)).rstrip("0").rstrip(".") + "K"
            text_tokens[index] = formatted_token
        elif len(text_tokens) > index + 1 and text_tokens[index + 1].lower() == "thousand":
            formatted_token = str(numeric_token).rstrip("0").rstrip(".") + "K"
            text_tokens[index] = formatted_token
            del text_tokens[index + 1]
        elif 1000000 <= numeric_token < 1000000000:
            formatted_token = "{num:.3f}".format(num=numeric_token / 1000000).rstrip("0").rstrip(".") + "M"
            text_tokens[index] = formatted_token
        elif len(text_tokens) > index + 1 and text_tokens[index + 1].lower() == "million":
            formatted_token = str(numeric_token).rstrip("0").rstrip(".") + "M"
            text_tokens[index] = formatted_token
            del text_tokens[index + 1]
        elif 1000000000 <= numeric_token:
            formatted_token = "{num:.3f}".format(num=numeric_token / 1000000000).rstrip("0").rstrip(".") + "B"
            text_tokens[index] = formatted_token
        elif len(text_tokens) > index + 1 and text_tokens[index + 1].lower() == "billion":
            formatted_token = str(numeric_token).rstrip("0").rstrip(".") + "B"
            text_tokens[index] = formatted_token
            del text_tokens[index + 1]

        # parse percentage
        # any number signifying percentage will be shown as #%
        if len(text_tokens) > index + 1:
            lower_case_next_token = text_tokens[index + 1].lower()
            if lower_case_next_token == "%" or lower_case_next_token == "percent" \
                    or lower_case_next_token == "percentage":
                formatted_token = str(numeric_token).rstrip("0").rstrip(".") + "%"
                text_tokens[index] = formatted_token
                del text_tokens[index + 1]

    def parse_date(self, text_tokens, index):
        """
        this function calls the appropriate function to parse a date.
        :param text_tokens: list of tokens
        :param index: the index of the month or the index of the 'MM/DD/YY' token
        :return: - reduction of the index as a result of deletion of previous tokens
                   in some cases such as '15th of July' we want to delete '15' and 'of' and insert '7~15'
                   in this cases we should bring the index back
        """

        if text_tokens[index].lower() in self.months:
            return self.parse_date_according_to_month(text_tokens, index)

        if text_tokens[index].count("/") == 2:
            self.parse_date_slash(text_tokens, index)
            return 0

    def parse_date_according_to_month(self, text_tokens, index):
        """
            parsing date of format '15 of Jun' or '15th of June' etc. to 'MM~DD' format
            :return - reduction of the index as a result of deletion of previous tokens
                      in some cases such as '15th of July' we want to delete '15' and 'of' and insert '7~15'
                      in this cases we should bring the index back
        """

        if len(text_tokens) > index + 1 and text_tokens[index].lower() in self.months:
            if text_tokens[index + 1] in self.days:  # July 15th
                text_tokens[index] = self.months.get(text_tokens[index].lower()) + \
                                     "~" + self.days.get(text_tokens[index + 1])
                del text_tokens[index + 1]
            elif text_tokens[index + 1].isnumeric():  # July 15
                text_tokens[index] = self.months.get(text_tokens[index].lower()) + \
                                     "~" + str(int(text_tokens[index + 1]))
                del text_tokens[index + 1]
        elif index - 1 >= 0:
            if text_tokens[index - 1] in self.days:  # 15th July
                text_tokens[index] = self.months.get(text_tokens[index].lower()) + \
                                     "~" + self.days.get(text_tokens[index - 1])
                del text_tokens[index - 1]
                return 1
            elif text_tokens[index - 1].isnumeric():  # 15 July
                text_tokens[index] = self.months.get(text_tokens[index].lower()) + \
                                     "~" + str(int(text_tokens[index - 1]))
                del text_tokens[index - 1]
                return 1
            elif text_tokens[index - 1] == "of" and index - 2 >= 0 \
                    and text_tokens[index - 2] in self.days:  # 15th of July
                text_tokens[index] = self.months.get(text_tokens[index].lower()) + \
                                     "~" + self.days.get(text_tokens[index - 2])
                del text_tokens[index - 1]  # delete for
                del text_tokens[index - 1]  # delete 15th
                return 2

            elif text_tokens[index - 1] == "of" and text_tokens[index - 2].isnumeric():  # 15 of july
                text_tokens[index] = self.months.get(text_tokens[index].lower()) + \
                                     "~" + str(int(text_tokens[index - 2]))
                del text_tokens[index - 1]  # delete for
                del text_tokens[index - 1]  # delete 15
                return 2
        return 0

    def parse_date_slash(self, text_tokens, index):
        """
            parse date with slash 'MM/DD/YY'
            to ['MM~DD', 'YY']
            @:param - index of the 'MM/DD/YY' token
        """

        splitted_date = text_tokens[index].split("/")
        if len(splitted_date) == 3 and splitted_date[0].isnumeric() and splitted_date[1].isnumeric() \
                and splitted_date[2].isnumeric():
            if int(splitted_date[0]) in range(0, 13) and int(splitted_date[1]) in range(0, 32):
                text_tokens[index] = str(int(splitted_date[0])) + "~" + str(int(splitted_date[1]))
                text_tokens.insert(index + 1, splitted_date[2])

    def parse_fraction(self, text_tokens, index):
        """
            this function parses fraction according to given rules ['35', '3/4'] - > ['35 3/4']
        :param text_tokens:
        :param index:
        :return:
        """

        splited_fruction = text_tokens[index].split("/")
        if index - 1 > 0 and text_tokens[index - 1].isnumeric() and \
                splited_fruction[0].isnumeric and splited_fruction[1].isnumeric():
            text_tokens[index - 1] = text_tokens[index - 1] + " " + text_tokens[index]
            del text_tokens[index]
            return True
        else:
            return False

    def parse_entities(self, text_tokens, index, entities):

        """
        Identify possible entities in the document.
        A possible entity is any sequence of tokens starting with a capital letter
        :param text_tokens: list of tokens to be parsed
        :param index: index of current parsed token
        :param entities: dictionary of possible entities
        """
        current_token = text_tokens[index]
        entity = ""

        # find a sequence of terms with capital letters
        while index + 1 < len(text_tokens) and current_token[0].isupper():
            entity += current_token + " "
            index += 1
            current_token = text_tokens[index]
        entity.rstrip(" ")

        # add new possible entity to dictionary
        if entity != "":
            if entity not in entities:
                entities[entity] = 1
            else:
                entities[entity] += 1

    def parse_capital_letters(self, tokenized_text, term_dict):

        """
        Parses token according to capital letters rule.
        Ensures a uniform appearance of tokens - if a token only appears in capital form - record as upper case
        Else, record in lower case
        :param tokenized_text - list, list of parsed tokens
        :param term_dict - dictionary, record uniform token appearance according to rule in currently parsed document
        """

        index = 0
        while index < len(tokenized_text):

            token = tokenized_text[index]

            if token != '':

                # save token as upper case
                # save token as lower and upper case
                formatted_token_lower = token.lower()
                formatted_token_upper = token.upper()

                # Add token to term dictionary
                # In the dictionary keep the term_frequency
                # term_frequency - how many times the term appeared in the document
                # key indicates if term is capital or lower case

                # Check if first letter is a capital letter
                if token[0].isupper():
                    # check in which form the token appears in dictionary and update it accordingly
                    if formatted_token_upper not in term_dict and formatted_token_lower not in term_dict:
                        term_dict[formatted_token_upper] = 1
                    elif formatted_token_upper in term_dict:
                        term_dict[formatted_token_upper] += 1
                    else:  # formatted_token_lower in capitals
                        term_dict[formatted_token_lower] += 1

                # If current term is lower case change key to lower case
                else:
                    # check in which form the token appears in dictionary and update it accordingly
                    if formatted_token_upper not in term_dict and formatted_token_lower not in term_dict:
                        term_dict[formatted_token_lower] = 1
                    elif formatted_token_upper in term_dict:  # replace format of token from upper case to lower case
                        term_dict[formatted_token_lower] = term_dict[formatted_token_upper] + 1
                        term_dict.pop(formatted_token_upper, None)  # remove upper case form from the dictionary
                    else:  # formatted_token_lower in capitals
                        term_dict[formatted_token_lower] += 1

            index += 1

    def parse_sentence(self, text, entities=None, stemming=False):

        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text: string - text to be parsed
        :param entities: dictionary - record possible entities in currently parsed document
        :param stemming: boolean variable True - with stemming, False - without stemming
        :return: list of parsed tokens
        """

        text_tokens = word_tokenize(text)

        index = 0
        while index < len(text_tokens):

            if text_tokens[index].lower() not in self.stop_words\
                    and text_tokens[index] not in self.punctuation_to_remove\
                    and text_tokens[index].isascii():

                # removing unnecessary symbols
                text_tokens[index] = text_tokens[index].rstrip(self.symbols).lstrip(self.symbols)
                if text_tokens[index] == "":
                    del text_tokens[index]
                    continue

                if text_tokens[index] == '#':
                    if not self.parse_hashtag(text_tokens, index):
                        continue
                elif text_tokens[index] == '@':
                    self.parse_tagging(text_tokens, index)
                elif text_tokens[index] == 'https' or text_tokens[index] == 'http':
                    self.parse_url(text_tokens, index)
                    continue

                # parse numeric values
                elif self.is_float(text_tokens[index]):
                    self.parse_numeric_values(text_tokens, index)

                # parse dates
                elif text_tokens[index].lower() in self.months or \
                        text_tokens[index].count("/") == 2:
                    index -= self.parse_date(text_tokens, index)

                # parse fractions
                elif text_tokens[index].count("/") == 1:
                    if self.parse_fraction(text_tokens, index):
                        continue

                # parse entities
                # entity is every sequence of tokens starting with a capital letter \
                # and appearing at least twice in the entire corpus
                if index + 1 < len(text_tokens) and text_tokens[index][0].isupper() \
                        and text_tokens[index + 1][0].isupper():
                    self.parse_entities(text_tokens, index, entities)

                # apply stemmer if stemming is True
                if stemming and len(text_tokens[index]) > 0 and text_tokens[index][0] not in "@#":
                    after_stemming = self.token_stemmer.stem_term(text_tokens[index])
                    if after_stemming != '':
                        text_tokens[index] = after_stemming

                if len(text_tokens[index]) == 1:
                    del text_tokens[index]
                    continue

                index += 1
            else:
                if not text_tokens[index].isascii():
                    # token is not ascii
                    valid_token = ''
                    for char in text_tokens[index]:
                        if char.isascii():
                            valid_token += char  # separate valid token from the ascii symbol appended to him
                        else:
                            # parsing emoji
                            emoji = [*findall(char).values()]  # unpack single emoji token and put in list
                            if len(emoji) > 0 and emoji[0] not in text_tokens:
                                text_tokens.append(emoji[0])
                                if len(emoji[0].split()) > 1:
                                    # add to text tokens emojis such as: 'smiling face', 'smiling', 'face'
                                    for emoji_token in emoji[0].split():
                                        text_tokens.append(emoji_token)

                    if valid_token != '':  # append the valid toke
                        text_tokens[index] = valid_token

                        # apply stemmer if stemming is True
                        if stemming and valid_token[0] not in "@#":
                            after_stemming = self.token_stemmer.stem_term(valid_token)
                            if after_stemming != '':
                                text_tokens[index] = after_stemming

                    else:
                        del text_tokens[index]  # not ascii symbols that we want to delete
                else:
                    del text_tokens[index]  # RT or punctuation that is in ascii

            if index > 0 and text_tokens[index - 1] == '':
                del text_tokens[index]

        return text_tokens

    def prep_url(self, url):
        """
            remove unnecessary signs from urls and not meaningful digits and letters
        """
        trans_table = url.maketrans("\\/|=<>.?%-:_", "            ")
        parsed_url = url.translate(trans_table)
        parsed_url_tokens = parsed_url.split()
        token_index = 0
        while token_index < len(parsed_url_tokens):
            if parsed_url_tokens[token_index].isdigit() \
                    or len(parsed_url_tokens[token_index]) == 1 \
                    or parsed_url_tokens[token_index] in ["www", "co", "com", "twitter", "status", "web", "https",
                                                          "http"]:
                parsed_url_tokens.remove(parsed_url_tokens[token_index])
                continue
            token_index += 1

        return " ".join(parsed_url_tokens)

    def remove_shortened_urls(self, full_text):
        try:
            full_text.index("https")
            return " ".join(filter(lambda splitted: splitted[:5] != 'https', full_text.split()))
        except ValueError:
            return full_text

    def parse_doc(self, doc_as_list, stemming=False):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-presenting the tweet.
        :return: Document object with corresponding fields.
        """

        url = ""
        retweet_url = ""
        quote_url = ""
        tweet_id = doc_as_list[0]
        tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        full_text = self.remove_shortened_urls(full_text)
        if doc_as_list[3] and doc_as_list[3] != "":
            url = self.get_valid_url(doc_as_list[3])
            url = self.prep_url(url)
        retweet_text = doc_as_list[5]
        if doc_as_list[6] and doc_as_list[6] != "":
            retweet_url = self.get_valid_url(doc_as_list[6])
            retweet_url = self.prep_url(retweet_url)
        quote_text = doc_as_list[8]
        if quote_text:
            quote_text = self.remove_shortened_urls(quote_text)
        else:
            quote_text = ""
        if doc_as_list[9] and doc_as_list[9] != "":
            quote_url = self.get_valid_url(doc_as_list[9])
            quote_url = self.prep_url(quote_url)
        term_dict = {}

        # dictionary for holding possible entities
        entities = dict()

        pre_processed_text = full_text + " " + quote_text + " " + url + " " + retweet_url + " " + quote_url
        tokenized_text = self.parse_sentence(pre_processed_text, entities, stemming)

        doc_length = len(tokenized_text)  # after text operations.

        # parse token by lower or upper case rule
        # parsing will build the term dictionary in a uniform upper/lower form and calculate the term frequency
        self.parse_capital_letters(tokenized_text, term_dict)

        max_tf = 0
        for tf in term_dict.values():
            if tf > max_tf:
                max_tf = tf

        unique_term_number = len(term_dict.keys())

        document = Document(tweet_id, tweet_date, full_text, url, retweet_text, retweet_url, quote_text,
                            quote_url, term_dict, doc_length, tweet_date, unique_term_number, entities, max_tf)
        return document
