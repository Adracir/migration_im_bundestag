# Die Entwicklung des Migrationsdiskurses im Deutschen Bundestag: Eine Analyse anhand von diachronen Word Embeddings
## Code for Master Thesis written by Ricarda Boente
#### Informationsverarbeitung MA, Universität zu Köln

### What is this work about?
This work is performing an experiment on the Parliamentary Protocols of the Deutscher Bundestag and analyzing keywords 
regarding migration with the use of Word2Vec word embeddings.
Further information on the interpretation and theoretical background of this code can be found in the written thesis with the same title.

### How is the code structured? How can it be used?
All necessary requirements to run the code can be installed using ```pip install -r requirements.txt```
It is recommended to use Python Version ```3.10```.

- ```main.py```: follow the workflow and call necessary methods from other files
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
    2. epochs 1-8 plain debate text extracted as txt files (```epoch<1-8>.txt```)
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
  - ```keywords.csv```: contains all keywords that can be analyzed in this experiment, including info on combinations of 
  different written forms (e.g. "Asylmissbrauch" and "Asylmißbrauch")
  - ```keywords_merged.csv```: contains all keywords, see above, plus information on their occurrences (only available 
  after frequency analysis)
  - ```keyword_comparing.csv```: groups keywords together that can be interesting to be analyzed in comparison to each 
  other
  - ```expected_values.csv```: stores information on which frequencies and sentiment values are expected for each 
  keyword and epoch
  - ```expected_freq_translation.csv```: groups expected frequency symbols (1-8) with their written forms 
  (häufig, selten ecc.)
  - ```expected_senti_translation.csv```: same for expected sentiment symbols (-1, 0.25, o+, ...) and their written 
  forms (sehr negativ, positiv, ...)
  - ```political_sentiwords.csv```: sentiment wordset used for the WEAT test, containing political words 
  from Walter et al. (2021)
  - ```sentiwords.csv```: basic sentiment wordset from Walter et al. (2021)
- ```results``` folder:
  - ```plots``` contains all visualizations of the calculated data on frequency, sentiment and word association level 
  - ```freqs.csv```: the measured frequency of each keyword in each epoch, total count and relative pMW value
  - ```expected_freq_results_slices.csv```: groups the frequency values into 8 slices that contain an equal amount of 
  measured values, giving a comparison frame for the data
  - ```kw_occurrences.csv```: for each keyword, when does it first and last occur? are epochs skipped? This info is 
  combined with ```keywords.csv``` to ```keywords_merged.csv``` and useful in many places
  - ```senti.csv```: for each keyword in each epoch for each different sentiword base set: WEAT values
  - ```expected_senti_results_slices.csv```: groups the sentiment values into 9 slices, giving a comparison frame for 
  the data  
  - ```nearest_neighbors.csv```: for each keyword in each epoch: the ten most similar words. (models not aligned)
  - ```nearest_neighbors_aligned.csv```: the same as above, but using the aligned models
  - ```aggregated_nearest_neighbors.csv```: contains 20 most similar words over all epochs for each keyword, including 
  their cosine similarity (taking the sum if a word appears in more than one epoch). 
  Retrieved using nearest_neighbors.csv

### Literature
- Gurevych, I. (2005). German Relatedness Datasets [dataset]. 
https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2440
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic 
Change. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
1489–1501. https://doi.org/10.18653/v1/P16-1141
- Walter, T., Kirschner, C., Eger, S., Glavaš, G., Lauscher, A., & Ponzetto, S. P. (2021). Diachronic Analysis of German
Parliamentary Proceedings: Ideological Shifts through the Lens of Political Biases. 2021 ACM/IEEE Joint Conference on 
Digital Libraries (JCDL), 51–60. https://doi.org/10.1109/JCDL52503.2021.00017


