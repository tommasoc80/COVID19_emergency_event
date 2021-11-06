# COVID19_emergency_event

Repository for the [EXCEPTIUS project](https://exceptius.com): Exceptional powers in times of SARS-COV-2.

This repository contains the data, scripts, and models that have been used to develop a multi-label, multilingual sentence classifier for the detection of exceptional measures in legislative documents.

The folder /annotations/ contains the manually annotated data per country, already split in Train/Dev/Test. The split is the one that has been used to run all experiments.

The folder /model/ contains the scripts used to train/test the different models (SVM, MLP, bi-GRU, XLM-RoBERTa)

The best fine-tuned XLM-RoBERTa model (further trained with MLM) is available here https://drive.google.com/drive/folders/1u2XGrwhImoLML9t8SCta6B3jEOwJ6Fxx?usp=sharing

All data (raw documents, pre-processed with UDPipe, and automatically annotated with XLM-RoBERTa for exceptional measures) are available in the DataverseNL platform (link coming soon).


# Citation
If you find any of the data or models useful, please cite the following paper (available [here](https://aclanthology.org/2021.nllp-1.5)): 

```
@inproceedings{tziafas-etal-2021-multilingual,
    title = "A Multilingual Approach to Identify and Classify Exceptional Measures against {COVID}-19",
    author = "Tziafas, Georgios  and
      de Saint-Phalle, Eugenie  and
      de Vries, Wietse  and
      Egger, Clara  and
      Caselli, Tommaso",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.5",
    pages = "46--62",
    abstract = "The COVID-19 pandemic has witnessed the implementations of exceptional measures by governments across the world to counteract its impact. This work presents the initial results of an on-going project, EXCEPTIUS, aiming to automatically identify, classify and com- pare exceptional measures against COVID-19 across 32 countries in Europe. To this goal, we created a corpus of legal documents with sentence-level annotations of eight different classes of exceptional measures that are im- plemented across these countries. We evalu- ated multiple multi-label classifiers on a manu- ally annotated corpus at sentence level. The XLM-RoBERTa model achieves highest per- formance on this multilingual multi-label clas- sification task, with a macro-average F1 score of 59.8{\%}.",
}
``` 







