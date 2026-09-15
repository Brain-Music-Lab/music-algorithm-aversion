import argparse

from .exc import ParameterError
from .arm_cleaning import main as arm_cleaning_main
from .context_source_comparison import main as context_main
from .countvectorizer import main as countvec_main
from .decision_tree import main as desc_tree_main
from .music_tfidf_vectorizer import main as tfidf_main
from .sent_analysis2 import main as sent_main


def main(args=None):
    parser = argparse.ArgumentParser("run")
    parser.add_argument("module")
    parser.parse_args(args)

    possible_modules = ['arm_cleaning', 'context_source_comparison', 'count_vectorizer', 'decision_tree', 'tfidf_vectorizer', 'sentiment_analysis']

    args = parser.parse_args(args)

    if args.module == 'arm_cleaning':
        arm_cleaning_main()
    elif args.module == 'context_source_comparison':
        context_main()
    elif args.module == "count_vectorizer":
        countvec_main()
    elif args.module == "tfidf_vectorizer":
        tfidf_main()
    elif args.module == "decision_tree":
        desc_tree_main()
    elif args.module == "sentiment_analysis":
        sent_main()
    else:
        raise ParameterError(f"Module not found. Possible modules are: {", ".join(possible_modules)}")


if __name__ == "__main__":
    main()
