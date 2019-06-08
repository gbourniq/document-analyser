import os
import time

import pytesseract

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim import corpora
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from parameters import search_word, topN
from wordcloud import WordCloud


def allowed_file(filename):
    """
    Check that uploaded files are either .pdf or .txt
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['pdf', 'txt'])


def cleanup_folder(folder_name):
    """
    Delete all content inside given folder
    """

    for filename in os.listdir(folder_name):
        os.remove(folder_name + "/" + filename)


def get_file_names_string(directory, processed_pdf_names):
    """
    Generate list of all uploaded file names (both .pdf and .txt)
    and apply html formatting
    """

    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filename_base = filename.split(".")[0]
            if filename_base in processed_pdf_names:
                filenames.append(filename_base+".pdf")
            else:
                filenames.append(filename_base+".txt")
    
    filenames.sort()
    list_file_names = "<br>• ".join(filenames)
    list_file_names = "• " + list_file_names
    return list_file_names


def load_to_dict(directory, processed_pdf_names):
    """
    Load text files into a dictionary with the following structure:
    data_dict = {
        "document1.txt":["sentence1", "sentence2", ..., "sentence n"],
        "document2.txt":["sentence1", "sentence2", ..., "sentence n"]
    }
    """
    
    filenames = []
    contents = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):

            filename_base = filename.split(".")[0]
            if filename_base in processed_pdf_names:
                filenames.append(filename_base+".pdf")
            else:
                filenames.append(filename_base+".txt")

            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf8") as f:
                doc_content = f.read().replace('\n', ' ').replace("U.S", "US")
                contents.append(doc_content)

    data = dict(zip(filenames, contents))
    
    # full text to sentences
    for doc_name, doc_content in data.items():
        # bloc of text to list of sentences contained in the dictionary
        data[doc_name] = sent_tokenize(doc_content)
        
    return data


def sentences_to_words(sentences):
    """
    Tokenize a list of sentences into a list of words
    """

    tokenizer = RegexpTokenizer(r'\w+')
    words = []
    words.append(tokenizer.tokenize((' '.join(sentences))))
    # Flatten list
    words = [word for sentence in words for word in sentence]
    return words


def concatenate_all_sentences(dictionary):
    """
    Extract sentences for all documents from the data dictionary
    and outputs a list of all sentences
    """

    sentences = []
    for doc_name, doc_content in dictionary.items():    
        sentences += doc_content
    return sentences


def words_to_lower(words):
    """
    Apply small capitalisation function (.lower) to all words
    """

    words_lower = []
    for word in words:
        if word == "US":
            # keep it as a noun (lowercase is detected as pronoun)
            words_lower.append(word)
        else:
            words_lower.append(word.lower())
    return words_lower


def remove_stop_words(words_lower):
    """
    remove stop words from a list of word
    """

    stop_words = stopwords.words('english')
    words_stopfree = " ".join([i for i in words_lower if i not in stop_words])
    return words_stopfree.split()


def words_to_lemma(words):
    """
    lemmatize a list of words to achieve the following :
    - Remove verbs
    - Remove cardinal digits, adverbs, modals, adjectives, determiners, prepositions, pronouns
    """

    words_lemma = []
    lemmatizer = WordNetLemmatizer()

    for word in pos_tag(words):
        if word[1][0] == "V":
            # word_lemma = lemmatizer.lemmatize(word[0], pos="v")
            # words_lemma.append(word_lemma)
            pass
        elif word[1] in ["CD", "RB", "MD", "DT", "IN", "PRP"]:
            pass
        else:
            word_lemma = lemmatizer.lemmatize(word[0])
            words_lemma.append(word_lemma)
    return words_lemma


def preprocess_docs(data):
    """
    Apply pre-processing functions to all documents
    """

    preprocessed_docs = []

    for doc_name, doc_content in data.items():
        words = sentences_to_words(doc_content)
        words_lower = words_to_lower(words)
        words_stopfree = remove_stop_words(words_lower)
        words_lemma = words_to_lemma(words_stopfree)    
        preprocessed_doc_string = " ".join(words_lemma)
        preprocessed_docs.append(preprocessed_doc_string)
    return preprocessed_docs


def find_most_common_words(preprocessed_docs, topN):
    """
    Returns the most common words and their frequency
    """

    words_clean = []
    for preprocessed_text in preprocessed_docs:
        words_clean = words_clean + preprocessed_text.split()

    freq = FreqDist(words_clean)
    return freq.most_common(topN)


def plot_freq(topN_words, folder_save):
    """
    Generates a bar plot to visualise most common words
    """

    most_common_words = [topic[0] for topic in topN_words]
    word_counts = [topic[1] for topic in topN_words]

    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 10))
    idx = np.arange(len(most_common_words))
    ax.bar(idx, word_counts, width=0.6, alpha=0.5)

    ax.set_xticks(idx)
    ax.set_xticklabels(most_common_words, rotation=45)
    ax.set_ylabel('Counts')

    plt.tight_layout(pad=0)
    plt.savefig(folder_save + "word_freq.png")
    plt.close()


def plot_word_freq(datadict, preprocessed_docs, topN_words, folder_save):
    """
    Generates a bar plot to visualise most common words split by documents
    """

    top_words_plot = []
    word_counts_in_docs = []
    doc_names = []

    for doc_name, doc_content in datadict.items():
        doc_names.append(doc_name)

    for i in range(len(preprocessed_docs)):
        word_counts_in_doc = [doc_names[i]]
        preprocess_doc_words = preprocessed_docs[i].split()
        freq = FreqDist(preprocess_doc_words).most_common()

        for word1 in topN_words:
            word_found_in_doc = False
            for word2 in freq:            
                if word1[0].lower() == word2[0].lower():
                    word_found_in_doc = True
                    word_counts_in_doc.append(word2[1])
                    break
            if word_found_in_doc is False:
                word_counts_in_doc.append(0)

        word_counts_in_docs.append(word_counts_in_doc)       
    top_words = [topic[0].title() for topic in topN_words]
    top_words_plot = ["Documents"] + top_words
    
    df = pd.DataFrame(columns=top_words_plot, data=word_counts_in_docs)
    
    sns.set()
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    bar_plot = df.set_index('Documents').T.plot(kind='bar',
                                                stacked=True,
                                                figsize=(25, 14),
                                                colormap='viridis',
                                                alpha=0.7,
                                                rot=45,
                                                fontsize=20).set_ylabel("Counts\n\n", fontsize=20)

    fig = bar_plot.get_figure()
    fig.savefig(folder_save + "word_freq.png", bbox_inches="tight")


def generate_word_cloud(word_cloud_string, folder_save):
    """
    Generate a word cloud image from a string
    """

    wordcloud = WordCloud(collocations=False, background_color='white', width=1600, height=800)
    wordcloud.generate(word_cloud_string)
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(folder_save + "wordcloud.png", bbox_inches="tight")
    plt.close()


def search_sentences(datadict, doc_name, search_word):
    """
    Returns sentences containing a given word, from a document
    """

    extracted_sentences = ""
    word_counts = 0
    
    for sentence in datadict[doc_name]:
        extracted_words = []
        if search_word.lower() in sentence.lower().split():
            for word in sentence.split():
                if search_word.lower() == word.lower():
                    word = "<b>"+word+"</b>"
                    extracted_words.append(word)
                    word_counts += 1
                else:
                    extracted_words.append(word)
            extracted_sentences += " ".join(extracted_words)+"<br>"
    extracted_sentences += "<br><br>"
    return extracted_sentences, word_counts


def get_synonyms(word):
    """
    Returns list of synonyms for a given word
    """

    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name().replace("_", " ").title())
    return ", ".join(list(set(synonyms))[:5])


def most_common_words_df(data, topN_words):
    """
    Generate a dataframe with the following fields:
    - Most common words
    - Document name
    - Word occurences
    - Sentences containing the word
    """

    words = []
    results = []

    for word in topN_words:
        for doc_name, doc_content in data.items():
            sentences_found, word_counts = search_sentences(data, doc_name, word[0])
            if word_counts > 0:
                title = f"<b>{word[0].title()}</b> found in <b>{doc_name}</b> ({word_counts} occurences)</b>"
                results.append(title)
                results.append(sentences_found)
                words.append(word[0].title())
                words.append(word[0].title())

    results = {
        "words" : words,
        "Full results": results
    }

    return pd.DataFrame.from_dict(results)


def search_word_df(data, search_word):
    """
    Generate a dataframe with the following fields:
    - Searched word (user input)
    - Document names
    - Word occurences
    - Sentences containing the word
    """

    common_words = []
    words_counts = []
    doc_names = []
    synonyms = []
    sentences = []

    for doc_name, doc_content in data.items():
        sentences_found, word_counts = search_sentences(data, doc_name, search_word)
        if word_counts > 0:
            common_words.append(search_word)
            doc_names.append(doc_name)
            synonyms.append(get_synonyms(search_word))
            sentences.append(sentences_found)
            words_counts.append(word_counts)

    if len(words_counts) == 0:
            common_words.append("Not found")
            doc_names.append("Not found")
            synonyms.append("Not found")
            sentences.append("Not found")
            words_counts.append("Not found")
            
    results = {
        "Word": common_words,
        "Document": doc_names,
        "Count": words_counts,
        "Synonyms": synonyms,
        "Sentences": sentences
    }

    return pd.DataFrame.from_dict(results) 


def find_topics(preprocessed_docs, num_topics=4, num_words=5, passes=50):
    """
    Returns main document topics in the form of relevant keywords
    """

    def generate_topics_df(topics):    
        topic_names = []
        topic_words = []
        results = []

        for topic in topics:
            words = ""
            for i in range(num_words):
                words = words + ", " + topic[1][i][0].capitalize() 
            topic_words.append(words[2:])

        for i in range(num_topics):
            topic_names.append("<b>Topic " + str(i+1) + "</b>")

        # Merge results
        for i in range(len(topic_names)):
            results.append(topic_names[i])
            results.append(topic_words[i]+"<br>")

        df_dict = {
            "Topics": topic_names,
            "Words": topic_words
        }

        return pd.DataFrame.from_dict(df_dict)

    docs = [d.split() for d in preprocessed_docs]

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(docs)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes)
    topics = ldamodel.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=False)
    
    return generate_topics_df(topics)


if __name__ == "__main__":
    
    print("Running functions.py - TEST")
    start_time = time.time()

    data = load_to_dict(r"C:\Users\guillaume.bournique\OneDrive - Avanade\Scripts_Jupyter\Text Analyser\docs", ["scandocument.pdf"])
    # print(data)

    preprocessed_docs = preprocess_docs(data)
    # print(preprocessed_docs)

    most_common_words = find_most_common_words(preprocessed_docs, topN)
    # print(most_common_words)

    plot_word_freq(data, preprocessed_docs, most_common_words, "uploads/")

    generate_word_cloud(" ".join(preprocessed_docs), "uploads/")

    common_words_df = most_common_words_df(data, most_common_words)
    # print(common_words_df)

    searched_word_df = search_word_df(data, search_word)
    # print(searched_word_df)

    topics_df = find_topics(preprocessed_docs, num_topics=num_topics, num_words=num_words, passes=50)
    # print(topics_df)

    print(f"Execution Time : {round(time.time() - start_time, 2)} secs")
