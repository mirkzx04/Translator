import re


class CustomTokenizer:
    def __init__(self):
        """
        Inizializza il Tokenizer

        PAD -> Per il padding
        SOS -> Start of Sentence
        EOS -> End of Sentence
        UNK -> Unknow word

        word2index mappa da parola a token
        index2word mappa da token a parola
        """
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        self.num_words = 4
        self.max_lenght = 0

        self.word2index = {
            '<PAD>' : self.PAD_token,
            '<SOS>' : self.SOS_token,
            '<EOS>' : self.EOS_token,
            '<UNK>' : self.UNK_token
        }
        self.index2word = {
            self.PAD_token : '<PAD>',
            self.SOS_token : '<SOS>',
            self.EOS_token : '<EOS>',
            self.UNK_token : '<UNK>'
        }

        self.count_word = {}

    def build_vocab(self, language, dataset):
        """
        Costruisce il vocabolario pre processando la frase prima
        """
        max_lengt = 0

        for sentence in dataset[language][1:]:
            sentence = sentence.lower()
            sentence = re.sub(r'[^\w\s]', '', sentence)

            self.add_sentence(sentence)

            sentence_len = len(sentence)

            self.max_lenght = max(self.max_lenght, sentence_len)
    
    def add_sentence(self, sentence):
        """
        Rimuove gli spazi dalle frasi e restituisce le parole
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Verifica se una parola è stata tokenizzata oppure no
        """
        if word not in self.word2index:
            self.num_words += 1
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.count_word[word] = 0
        else:
            self.count_word[word] += 1

    def tokenize_sentence(self, sentence):
        """
        Restituisce la lista di token di una frase
        """
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)

        words = sentence.split(' ')
        tokens = [self.SOS_token]

        for word in words:
            if word in self.word2index:
                tokens.append(self.get_word_index(word))
            else:
                tokens.append(self.UNK_token)

        tokens.append(self.EOS_token)

        if len(tokens) < self.max_lenght:
            tokens.extend([self.PAD_token] * (self.max_lenght - len(tokens)))

        return tokens
    
    def detokenize_sentence(self, tokens):
        """
        Restituisce la frase rispetto i token in tokens
        """
        sentence = ''
        for token in tokens:
            if token == self.SOS_token or token == self.PAD_token:
                continue
            if token == self.EOS_tokentoken:
                break

            sentence += f'{self.get_index_word(token)}'

    def get_word_index(self, word):
        return self.word2index[word]
    
    def get_index_word(self, index):
        return self.index2word[index]
    
