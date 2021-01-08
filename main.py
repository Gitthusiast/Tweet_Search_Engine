import search_engine_best
import os
from glob import glob
import utils
import matplotlib.pyplot as plt
import pandas as pd
from os.path import splitext as os_path_splitext


def report_analysis(output_path, stemming, engine):

    with open(os.path.join(output_path, "log.txt"), "a") as log:
        log.write("with stemming:\n" if stemming else "without stemming:\n")

        # number of unique terms in the index
        stemmed = "WithStem\\" if stemming else "WithoutStem\\"
        inverted_idx, posting = engine.load_index(os.path.join(output_path, stemmed + "idx_bench"))
        term_number = len(inverted_idx.keys())
        log.write("Numbers of terms in the index {ifStem} is: {num}\n".format(
            num=term_number, ifStem="with stemming" if stemming else "without stemming"))

        # top 10 max, min by total frequency
        sorted_index = [[entry[0], entry[1][1]] for entry in inverted_idx.items()]
        sorted_index.sort(key=lambda entry: entry[1], reverse=True)
        ranked_index = [[entry[0], entry[1], i] for i, entry in enumerate(sorted_index, 1)]

        log.write(" top 10 max: {}\n".format(ranked_index[:10]))
        log.write(" bottom 10 min: {}\n".format(ranked_index[-10:]))

        # zipf
        rank = [entry[2] for entry in ranked_index]  # x
        frequency = [entry[1] for entry in ranked_index]  # y

        plt.plot(rank, frequency, "-ok")
        plt.xlabel("rank")
        plt.ylabel("frequency")
        plt.title("Zipf law in twitter corpus")
        plt.savefig("Zipf law in twitter corpus.png")


def main():
    bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
    bench_lbls_path = os.path.join('data', 'benchmark_lbls_train.csv')
    queries_path = os.path.join('data', 'queries_train.tsv')
    output_path = os.getcwd() + "\\results analysis"

    bench_lbls = pd.read_csv(bench_lbls_path,
                             dtype={'query': int, 'tweet': str, 'y_true': int})
    q2n_relevant = bench_lbls.groupby('query')['y_true'].sum().to_dict()
    queries = pd.read_csv(queries_path, sep='\t')

    stemming = True
    engine = search_engine_best.SearchEngine()
    stemmed = "WithStem\\" if stemming else "WithoutStem\\"
    engine.build_index_from_parquet(bench_data_path, toSave=True, save_path=os.path.join(output_path, stemmed + "idx_bench"))

    with open(os.path.join(output_path, "log.txt"), "a") as log:
        log.write("Corpus size {ifStem}: {num}\n".format(
            num=engine._indexer.number_of_documents, ifStem="with stemming" if stemming else "without stemming"))

        log.write("Avrgdl size {ifStem}: {num}\n".format(
            num=engine._indexer.average_document_length, ifStem="with stemming" if stemming else "without stemming"))

    report_analysis(output_path, stemming, engine)


if __name__ == '__main__':
    main()
