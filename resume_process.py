from bag_of_words import *
from word_embedding import *

def process_resume(input_file, dataset_difficulty):
    '''the first output is the bag dataset, the second output is the embedding dataset. 
    the left input is the file name, the right input is the "difficult" / "easy" 
    (the number of columns differ for "difficult" and "easy" bags i believe
    then the last line is just deciphering the predicted output'''
    
    resume_file = open(input_file, "r")
    resume = resume_file.read()
    resume_file.close()
    
    # should output 116 columns
    if dataset_difficulty == "easy":
        top_stems = [x[0] for x in pd.read_csv("top_stems.csv", header=None).values.tolist()]
        tokens = get_tokenized_text(resume)
        is_noun_or_adj = get_nouns_and_adjs(tokens)
        lemmatized_tokens = pd.Series([[lemmatize(word) for word in tokens]], dtype="object")
        X_bag = pd.DataFrame(get_tfidf(lemmatized_tokens, top_stems), columns=top_stems)
    
    # should output 100 columns
    else:
        top_stems = [x[0] for x in pd.read_csv("top_stems.csv", header=None).values.tolist()]
        tokens = get_tokenized_text(resume)
        is_noun_or_adj = get_nouns_and_adjs(tokens)
        lemmatized_tokens = pd.Series([[lemmatize(word) for word in tokens]], dtype="object")
        X_bag = pd.DataFrame(get_tfidf(lemmatized_tokens, top_stems), columns=top_stems)
    
    X_embedding = pd.DataFrame(embed(resume)).T
    
    return X_bag, X_embedding