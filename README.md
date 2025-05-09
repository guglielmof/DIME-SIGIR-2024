# DIME

This repository includes the code to replicate the results of the paper ["Dimension Importance Estimation for Dense Information Retrieval"](https://dl.acm.org/doi/pdf/10.1145/3626772.3657691) by Guglielmo Faggioli,  Nicola Ferro, Raffaele Perego, Nicola Tonellotto, presented at SIGIR 2025.


## Prerequisites

Besides the requirements in the `requirements.txt` file, also faiss should be installed. Follow the instructions available [here](https://github.com/facebookresearch/faiss?tab=readme-ov-file) to install it.

To work, the code requires the encoding of the corpora to be already available in a memmap. A memmap is a numpy data structure used to store and read large numpy matrices efficiently. More details are available [here](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

 To compute the encodings, it is possible to use the method `encode_documents` of the instances of the classes available in the `ir_models.dense` module. 
These memmaps must be saved in:

    data/memmap/<name_of_the_corpus>/<name_of_the_encoder>

Within this directory, there should be two files `<name_of_the_encoder>.dat` that contains the mammap and `<name_of_the_encoder>_map.csv` that should be a csv with two columns `and `offset`. The first column is the document id in the corpus, while the second is the offset (i.e., the number of the row) at which the document is encoded in the memmap.

to replicate exactly the results of the paper, `<name_of_the_corpus>`should be one between `msmarco-passages`and `tipster`. While `<name_of_the_encoder>` is one among `ance, contriever, tasb`.

## Running the code
If you satisfy the prerequisites (i.e., you have a memmap data structure that stores the encoding of all the documents in the corpus in the directory, as described above), the code can run from the main directory with the following line:

    python code/main.py -c <name_of_the_collection> -e <name_of_the_encoder> -d <name_of_the_dime>

Name of the collection should be one among `trec-dl-2019`, `trec-dl-2020`,  `trec-robust-2004`.
Concerning the name of the dime, it should be one among `oracle`, `rel`, `llm`, `prf`.


### Citing this work
To cite this work, use the following bib entry:

    @inproceedings{DBLP:conf/sigir/Faggioli00T24,
        author       = {Guglielmo Faggioli and
                  Nicola Ferro and
                  Raffaele Perego and
                  Nicola Tonellotto},
        editor       = {Grace Hui Yang and
                  Hongning Wang and
                  Sam Han and
                  Claudia Hauff and
                  Guido Zuccon and
                  Yi Zhang},
        title        = {Dimension Importance Estimation for Dense Information Retrieval},
        booktitle    = {Proceedings of the 47th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2024, Washington
                  DC, USA, July 14-18, 2024},
        pages        = {1318--1328},
        publisher    = {{ACM}},
        year         = {2024},
        url          = {https://doi.org/10.1145/3626772.3657691},
        doi          = {10.1145/3626772.3657691},
    }
  

