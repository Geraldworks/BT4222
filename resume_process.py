from bag_of_words import *
from word_embedding import *

def process_resume(input_file, dataset_difficulty):
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
    if dataset_difficulty == "difficult":
        top_stems = [x[0] for x in pd.read_csv("top_stems.csv", header=None).values.tolist()]
        tokens = get_tokenized_text(resume)
        is_noun_or_adj = get_nouns_and_adjs(tokens)
        lemmatized_tokens = pd.Series([[lemmatize(word) for word in tokens]], dtype="object")
        X_bag = pd.DataFrame(get_tfidf(lemmatized_tokens, top_stems), columns=top_stems)
    
    X_embedding = pd.DataFrame(embed(resume)).T
    
    return X_bag, X_embedding