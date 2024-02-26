# Die Entwicklung des Migrationsdiskurses im Deutschen Bundestag: Eine Analyse anhand von diachronen Word Embeddings
## Code for Master Thesis written by Ricarda Boente
#### Informationsverarbeitung MA, Universität zu Köln

### What is this work about?
This work contains the code used to perform an experiment on the Parliamentary Protocols of the German Bundestag,
analyzing keywords regarding migration with the use of Word2Vec word embeddings.
Further information on the interpretation and theoretical background of this code can be found in the written thesis 
with the same title (PDF file in german language).

### How is the code structured? How can it be used?
All necessary requirements to run the code can be installed using ```pip install -r requirements.txt```. 
It is recommended to use Python Version ```3.10```.

- ```main.py```: follow the workflow and execute methods from other files
- ```prepare_corpus.py```: transform the plenary protocol texts from xml files to nested lists (tokenized, lemmatized, 
cleaned and sorted by time epochs needed for the experiment)
- ```embeddings.py```: train Word2Vec embeddings for each epoch. Align the models according to Hamilton et al. (2016).
- ```experiment.py```: execute the experiment by calculating data on frequency, sentiment (WEAT) and word associations 
(nearest neighbors) for the keywords
- ```visualizations.py```: visualize the data
- ```utils.py```: some helper methods
### Data in this project
- ```data``` folder:
  - ```corpus```
    1. plenary protocols as xml files downloaded from https://www.bundestag.de/services/opendata, structured by folders per 
    election period (```wp1```-```wp20```). Omitted in GitHub version, only example file contained in official version.
    2. plain debate text for epochs 1-8 extracted as txt files (```epoch<1-8>.txt```)
    3. prepared text as pickled nested lists, ready for embedding training (```epoch<1-8>_prepared_lemma```)
  - ```evaluation``` (omitted in GitHub version)
    - word pairs for embedding evaluation, downloaded from Gurevych (2005)
  - ```models``` (omitted in GitHub version)
    - ```aligned_models```
      - models aligned with different start epochs and loopholes
    - ```base_models```
      - basic Word2Vec models for the epochs
  - ```session_markers.csv```: regular expressions used to distinguish debate text from additional information in the 
  plenary protocol files before election period 20
  - ```epochs.csv```: list of epochs with their time range and written forms
  - ```keywords.csv```: all keywords to be analyzed in this experiment, including info on combinations of 
  different written forms (e.g. "Asylmissbrauch" and "Asylmißbrauch")
  - ```keywords_merged.csv```: all keywords, see above, plus information on their occurrences (only available 
  after frequency analysis)
  - ```keyword_comparing.csv```: keywords groups to be analyzed in comparison with each other
  - ```expected_values.csv```: expected frequency and sentiment values for each keyword in each epoch
  - ```expected_freq_translation.csv```: expected frequency symbols (1-8) with their written forms 
  ('häufig', 'selten' ecc.)
  - ```expected_senti_translation.csv```: same for expected sentiment symbols (-1, 0.25, o+, ...) and their written 
  forms ('sehr negativ', 'positiv', ...)
  - ```political_sentiwords.csv```: sentiment wordset used for the WEAT test, containing political words 
  from Walter et al. (2021)
  - ```sentiwords.csv```: basic sentiment wordset from Walter et al. (2021)
- ```results``` folder:
  - ```plots``` folder: all visualizations of the calculated data on frequency, sentiment and word association level 
  - ```freqs.csv```: the measured frequency of each keyword in each epoch, total count and relative pMW value
  - ```expected_freq_results_slices.csv```: 8 equally large groups of the frequency values, giving a comparison frame 
  - for the data
  - ```kw_occurrences.csv```: info on occurrence (count > 4) of each keyword (first epoch, last epoch, skipped epochs).
  combined with ```keywords.csv``` to ```keywords_merged.csv``` 
  - ```senti.csv```: WEAT values for each keyword in each epoch for each different sentiword base set
  - ```expected_senti_results_slices.csv```: 9 groups of the sentiment values, giving a comparison frame for 
  the data  
  - ```nearest_neighbors.csv```: the ten most cosine-similar words for each keyword in each epoch. (models not aligned)
  - ```nearest_neighbors_aligned.csv```: same as above, but using the aligned models
  - ```aggregated_nearest_neighbors.csv```: 20 most cosine-similar words, taking the sum over all epochs for 
  each keyword, including. Retrieved using nearest_neighbors.csv

### Literature
- Gurevych, I. (2005). German Relatedness Datasets [dataset]. 
https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2440
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic 
Change. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
1489–1501. https://doi.org/10.18653/v1/P16-1141
- Walter, T., Kirschner, C., Eger, S., Glavaš, G., Lauscher, A., & Ponzetto, S. P. (2021). Diachronic Analysis of German
Parliamentary Proceedings: Ideological Shifts through the Lens of Political Biases. 2021 ACM/IEEE Joint Conference on 
Digital Libraries (JCDL), 51–60. https://doi.org/10.1109/JCDL52503.2021.00017


