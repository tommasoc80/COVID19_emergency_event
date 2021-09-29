# COVID19_emergency_event

Repository for the [EXCEPTIUS project](https://exceptius.com): Exceptional powers in times of SARS-COV-2.

This repository contains the data, scripts, and models that have been used to develop a multi-label, multilingual sentence classifier for the detection of exceptional measures in legislative documents.

The folder /annotations/ contains the manually annotated data per country, already split in Train/Dev/Test. The split is the one that has been used to run all experiments.

The folder /model/ contains the scripts used to train/test the different models (SVM, MLP, bi-GRU, XLM-RoBERTa)

The best fine-tuned XLM-RoBERTa model (further trained with MLM) is available here https://drive.google.com/drive/folders/1u2XGrwhImoLML9t8SCta6B3jEOwJ6Fxx?usp=sharing

All data (raw documents, pre-processed with UDPipe, and automatically annotated with XLM-RoBERTa for exceptional measures) are available in the DataverseNL platform (link coming soon).


# Citation
If you find any of the data or models useful, please cite the following paper (publication link coming soon): 

```
@InProceedings{tziafas_et_al_2021,
  author    = {Tziafas, Georgios and de Saint-Phalle, Eugenie and de Vries, Wietse and Egger, Clara and Caselli, Tommaso},
  title     = {{A Multilingual Approach to Identify and Classify Exceptional Measures Against COVID-19}},
  booktitle = {Proceedings of the 3rd Natural Legal Language Processing (NLLP 2021)},
  year      = {2021}
}
``` 







