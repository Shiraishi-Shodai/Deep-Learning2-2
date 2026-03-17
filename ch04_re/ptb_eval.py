import pickle
import sys
sys.path.append("..")
from common.util import cos_similarity, most_similar
from common.trainer import Trainer


def main():
    queries = ["you", "year", "car", "toyota"]

    pkl_filename = "ptb_params.pkl"
    with open(pkl_filename, "rb") as f:
        params = pickle.load(f)
    

    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]
    word_vec = params["word_vec"]
    
    top = 5

    for query in queries:
        most_similar(query, word_to_id, id_to_word, word_matrix=word_vec, top=top)
    
if __name__ == "__main__":
    main()