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

    if dataset_difficulty == "easy":
        top_stem_filename = "top_stems_easy.csv"
    else:
        top_stem_filename = "top_stems_difficult.csv"
    top_stems = [x[0] for x in pd.read_csv(top_stem_filename, header=None).values]
    tokens = get_tokenized_text(resume)
    lemmatized_tokens = pd.Series([[lemmatize(word) for word in tokens]], dtype="object")
    X_bag = pd.DataFrame(get_tfidf(lemmatized_tokens, top_stems), columns=top_stems)
    
    X_embedding = pd.DataFrame(embed(resume)).T
    
    return X_bag, X_embedding