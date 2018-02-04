import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))
COUNTS = Counter(WORDS)

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def Pwords(words):
    "Probability of words, assuming each word is independent of others."
    return product(P(w) for w in words)

def product(nums):
    "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
    result = 1
    for x in nums:
        result *= x
    return result

def memo(f):
    "Memoize function f, whose args must all be hashable."
    cache = {}
    def fmemo(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    fmemo.cache = cache
    return fmemo

def splits(text, start=0, L=30):
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    return [(text[:i], text[i:])
            for i in range(start, min(len(text), L)+1)]

@memo
def segment(text):
    "Return a list of words that is the most probable segmentation of text."
    if not text:
        return []
    else:
        candidates = ([first] + segment(rest)
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=Pwords)

def segment_punc(s):
    '''insert space before and after punc'''

    # For words that use the ' char for contractions (won't, can't, you're, etc)
    # we want to keep the single quote character.

    # s = re.sub(r"[“”]", r'"', s)   # Convert to "
    # s = re.sub(r"[‘’]", r'\'', s)  # Convert to '

    # s = re.sub(r"([\'])(\w+)([\'])", r'\1 \2 \3', s)  # Replace 'word' with ' word '
    s = re.sub(r"(\W\'+)(\w+)", r'\1 \2', s)  # Replace 'word with ' word
    s = re.sub(r"(\w+)(\'+\W)", r'\1 \2', s)  # Replace word' with word '
    s = re.sub(r"(\w+)(\'+s)", r'\1 \'s', s)  # Replace word's with word 's
    s = re.sub(r"([!@#$%^&*\(\)+-=_?/:;,\"\.<>|\\\{\}\[\]–´~“”‘’]+)", r" \1 ", s)

    # s = re.sub(r"([!@#$%^&*\(\)+-=_?/:;,\"\'\.<>|\\\{\}\[\]–´~“”‘’]+)", r" \1 ", s)

    return s

def segment_curse_words(s):

    s_proc = re.sub(r"(fucking|fucked|fuckers*|fuck)", r" \1 ", s.lower())
    s_proc = re.sub(r"(f+u+c+k+e+r+s+)", r" fucker ", s_proc)
    s_proc = re.sub(r"(f+u+c+kk+)", r" fuck ", s_proc)
    s_proc = re.sub(r"(s+h+i+t+(?![aeiou]))", r" shit ", s_proc)
    s_proc = re.sub(r"fag+o+t+s*", r" faggot ", s_proc)
    s_proc = re.sub(r"^(f+a+g+s*)", r" fag ", s_proc)
    s_proc = re.sub(r"(f+a+g+)$", r" fag ", s_proc)
    s_proc = re.sub(r"(f+a+g+s+)$", r" fags ", s_proc)
    s_proc = re.sub(r"^(c+u+n+t*)", r" cunt ", s_proc)
    s_proc = re.sub(r"(c+u+n+t*)$", r" cunt ", s_proc)
    s_proc = re.sub(r"n+i+gg+e+r+", r" nigger ", s_proc)
    s_proc = re.sub(r"r+e+t+a+r+d+s*", r" retard ", s_proc)

    return s_proc


def text_to_wordlist(text,
                     remove_punc=True,
                     remove_numbers=True,
                     remove_stopwords=False,
                     remove_ip=False,
                     stem_words=False,
                     spell_correct=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Remove ip addresses
    if remove_ip:
        text = re.sub(r"[\d+\.]+[\d+]", " ", text)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    #Remove Special Characters
    if remove_punc:
        #Regex to remove all characters not in the regex compiler below
        #This will remove the "special characters"
        # special_character_removal=re.compile(r'[^a-z\d!?*#%$@&\[\]^_\(\)\.,\'\"\s]',re.IGNORECASE)
        special_character_removal=re.compile(r'[^a-z\d!?*\'\"\s]',re.IGNORECASE)
        text=special_character_removal.sub('',text)

    #Replace Numbers
    if remove_numbers:
        #regex to replace all numerics
        replace_numbers=re.compile(r'\d+',re.IGNORECASE)
        text=replace_numbers.sub('n',text)

    #Spell check
    if spell_correct:
        text = text.lower().split()
        text_corrected = []

        for w in text:
            if embeddings_index.get(w) is not None:
                text_corrected.append(w)

            else:
                corrected_word = correction(w)
                text_corrected.append(corrected_word)

        return ' '.join(text_corrected)

    #Stem words
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)
