import re
import os
import math
import string
string.punctuation
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from flask import Flask, request, render_template

# Global Variables
N = 20
"""doc_freq: contains document frequency of each term i.e. in how many documents a term occurs"""
doc_freq = {}

"""filenames: names of documents in corpus."""
filenames = []

"""dictionary: contains all unique terms in a corpus"""
dictionary  = set()

"""inverted_index: contains terms with the list of documents in which these terms occur as well as the frequency of occurence."""
inverted_index = {}

"""query_tfidf: tfidf value for terms appearing in query"""
query_tfidf = defaultdict(float)

"""tfidf: A defaultdict to store the tf-idf value for each document's terms. This helps in understanding 
the importance of a word in a document within the corpus."""
tfidf = defaultdict(float)

"""idf:  A defaultdict to store the inverse document frequency (idf) for each term. The idf is used to 
calculate the tf-idf. It helps in giving higher value to terms that occur in fewer documents, thus 
helping in distinguishing the relevance of documents.
"""
idf = defaultdict(float)

# Creating a new Flask web server instance
app = Flask(__name__)

# A list of common words (stop words) that are usually filtered out in natural language processing 
# because they occur so frequently that they carry little meaningful information.
stop_words = ['a', 'is', 'the', 'of', 'all', 'and', 'to', 'can', 'be', 'as', 'once', 'for',
             'at', 'am', 'are', 'has', 'have', 'had', 'up', 'his', 'her', 'in', 'on', 'no', 'we', 'do']


def main():
    # Initialize terms and their postings from the documents
    initialize_terms_and_postings()

    # Calculate the document frequency for each term
    initialize_document_frequency()

    # Calculate the tf-idf (term frequency-inverse document frequency) for each term
    tf_idf()

    # Uncomment the following lines if you want to enter search queries in a loop
    # while True:
    #     str = input("Search Query>>> ")  # Get the search query from the user
    #     search_query(str)  # Search the query in the documents

def preprocessing(text):
    # Nested function to clean the corpus
    def remove_punctuation(text):
        # Remove punctuation
        punctuation_free = ''.join([char for char in text if char not in string.punctuation])
        # Check for characters not in English
        english_characters = set(string.ascii_letters + string.digits + ' ')  # English letters, digits, and space
        filtered_text = ''.join([char if char in english_characters else ' ' for char in punctuation_free])
        return filtered_text

    def remove_numbers_from_token(token):
        result = re.sub(r'\d', '', token) #remove numbers
        return result

    def remove_single_alphabets_from_token(token):
        result = re.sub(r'\b[a-zA-Z]\b', '', token) #remove singlesingle alphabets
        return result

    def tokenize(text):
        # Create a reference variable for Class WordPunctTokenizer
        tk = WordPunctTokenizer()
        # Use tokenize method
        tokenlist = tk.tokenize(text)
        cleaned_tokens = [remove_numbers_from_token(token) for token in tokenlist]
        cleaned_tokens = [remove_single_alphabets_from_token(token) for token in cleaned_tokens]

        cleaned_tokens = list(filter(lambda x: x != '',cleaned_tokens ))

        cleaned_tokens = [token for token in cleaned_tokens if token.isalnum()]
        return cleaned_tokens

    def lower_the_tokens(words):
        lower = [word.lower() for word in words]
        return lower

    def remove_stopwords(list_of_words):
        proper_text = [word for word in list_of_words if word.lower() not in stop_words]
        return proper_text

    def porter_stemming(words):
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words]
        return stemmed_words

    # Apply all preprocessing steps
    processed_text = remove_punctuation(text)
    processed_tokens = tokenize(processed_text)
    processed_tokens = lower_the_tokens(processed_tokens)
    processed_tokens = remove_stopwords(processed_tokens)
    processed_tokens = porter_stemming(processed_tokens)

    return processed_tokens

def initialize_terms_and_postings():
    global dictionary, inverted_index, filenames
    # Check if the inverted index file already exists
    if not os.path.exists('inverted_indexA2.txt'):
        # Read files to make tokens
        for root, dirs, files in os.walk('ResearchPapers/'):
            for file_name in files:  # Document in Corpus
                f = file_name.split('.txt')
                filenames.append(f[0])
                
                with open(os.path.join(root, file_name), 'r') as file:  # Extracting sentences in a document
                    content = file.read()

                    current_tokens = preprocessing(content)
                    unique_terms = set(current_tokens)
                    dictionary.update(unique_terms)
                    term_freq = {}
                    for term in current_tokens:
                        term_freq[term] = term_freq.get(term, 0) + 1

                    # for term, freq in term_freq.items():
                    #     print(DocID,"  Is ", term , "  -> ",freq)

                    # Update inverted index
                    rot, _ = os.path.splitext(file_name)
                    DocID = int(rot)
                    for term, freq in term_freq.items():
                        if term not in inverted_index:
                            inverted_index[term] = {}
                        inverted_index[term][DocID] = freq

        # Write inverted index into a file
            with open('inverted_indexA2.txt', 'w') as file:
                for i, (term, doc_freqs) in enumerate(inverted_index.items()):
                    file.write(f"{i+1} - {term} -> ")
                    for doc_id, freq in doc_freqs.items():
                        file.write(f"({doc_id}: {freq}), ")
                    file.write("\n")
    else:
        with open('inverted_indexA2.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(' -> ')
                term = parts[0].split(' - ')[1]  # Extract the term
                postings_info = parts[1]

                # Extract document IDs and frequencies from postings_info
                doc_freq_list = postings_info.split(', ')
                for doc_freq in doc_freq_list:
                    doc_id, freq = doc_freq.strip('()').split(': ')
                    par = freq.split(')')
                    freq = par[0]
                    inverted_index.setdefault(term, {})
                    inverted_index[term][doc_id] = freq
                    # print(inverted_index[term][doc_id],end='\n')
                   
                    # If doc_id is not in filenames, add it
                    if doc_id not in filenames:
                        filenames.append(doc_id)

                # Add the term to the dictionary
                dictionary.add(term)
                # break
    return

def initialize_document_frequency():
    global doc_freq
    for term in dictionary:
        doc_freq[term] = len(inverted_index[term])
        # print(term," = ", doc_freq[term])
    
    return

def tf_idf():
    """Calculates TF-IDF value"""
    if not os.path.exists('TF-IDF.txt'):
        with open('TF-IDF.txt','w') as file:    
            for id in filenames:
                tfidf.setdefault(id, {})
                file.write(f'{id} -> ')
                for term in dictionary:
                    if id in inverted_index[term]:
                        tfidf[id][term] =  float(inverted_index[term][id])*inverse_document_frequency(term)
                        file.write(f'{term}: {tfidf[id][term]}, ')
                    else:
                        tfidf[id][term] = 0.0
                file.write('\n')
    else:
        with open('TF-IDF.txt', 'r') as file:
            for line in file:
                doc_id, terms = line.split(' -> ')
                doc_id = int(doc_id)
                tfidf[doc_id] = {}
                terms = terms.split(', ')
                for term in terms:
                    if term and ': ' in term:
                        term_name, value = term.split(': ')
                        term_name = term_name.strip()
                        value = value.strip()
                        tfidf[doc_id][term_name] = float(value)
                        # print(term, term_name, tfidf[doc_id][term_name],end='\n')

    return tfidf

def inverse_document_frequency(term):
    """Returns the inverse document frequency of term.  Note that if
    term isn't in the dictionary then it returns 0, by convention."""
    if term in dictionary:
        idf[term] =  math.log(N/doc_freq[term],10)
        return idf[term]
    else:
        return 0.0

def cosine_similarity(query, doc):
    # Calculate the dot product of the query and document vectors
    dot_product = sum(query.get(term, 0) * doc.get(term, 0) for term in set(query) & set(doc))
    
    # Calculate the norm (magnitude) of the query vector
    query_norm = math.sqrt(sum(val ** 2 for val in query.values()))
    
    # Calculate the norm (magnitude) of the document vector
    doc_norm = math.sqrt(sum(val ** 2 for val in doc.values()))
    
    # If either norm is zero, return zero to avoid division by zero
    if query_norm == 0 or doc_norm == 0:
        return 0
    
    # Return the cosine similarity, which is the dot product divided by the product of the norms
    return dot_product / (query_norm * doc_norm)

def search_query(query,alpha = 0.025, top_n=11):
    global query_tfidf
    query_tfidf = {} 
    # Split the query into terms
    query_terms = preprocessing(query)

    query_vector = {}
    """query_vector: contains Term-frequeny of query terms."""
    for token in query_terms:
        query_vector[token] = query_vector.get(token, 0) + 1

    """Calculate tf-idf weight for query_vector"""
    for term, freq in query_vector.items():
        if inverse_document_frequency(term) != 0.0:
            query_tfidf[term] = freq * idf[term]

    """Calculate similarity between documents in which query terms appear"""
    document_scores = {}
    for doc_id, doc_vector in tfidf.items():
        score = cosine_similarity(query_tfidf, doc_vector)
        if score >= alpha:                  # Only consider documents with a score >= alpha
            document_scores[doc_id] = score

     # Sort documents by score in descending order and return the top n documents
    top_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Optionally, print each document and its score
    for doc, score in top_documents:
        print(f"Document ID: {doc}, Score: {score}")

    return top_documents

@app.route('/', methods=['GET', 'POST'])
def search():
    # Check if the request method is POST
    if request.method == 'POST':
        # If it is, get the query from the form data
        query = request.form.get('query')
        # Search the query in the documents
        results = search_query(query)
        # Render the results page with the search results
        return render_template('results.html', results=results)
    # If the request method is not POST (i.e., it's GET), render the search page
    return render_template('search.html')

if __name__ == '__main__':
    main()
    app.run(debug=True)