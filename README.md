# gender bias
In this repo, I publish the modified code part bachelor thesis. The code could be used to develop a model to detect some gender bias in text.
The origin code was performed on schoolbook texts, which I do not have a license for. Therefore, I have to adopt the code before I could publish it.

## Input text
To analyze a text in the way I did in my thesis, the text must be larger, at least a few thousand sentences and of course has to written in German.

## Preprocessing
Before a model could be trained with the text, there must be some preprocessing steps.

    Make all words to token
    Lemmatization of the word token
    Replace all female, male and neutral names which gender token

Step 3 was me idea, normally at the end WEAT are done with a few female and male names, but I want to prevent a selection bias by myself.

Then clean the text, all artifacts and stop list words are removed.
The preprocessing could be done with the preprocessing.py file.

## Hyperparameters for the model
To find the best working hyperparameters, I generated a few hundred models. The results of the preprocessing will determine which hyperparameters are useful. If the minimum appearance is too high, not enough word token are in the model. For this step, use the generate_models.py file.

The models could be compared with the outcome of the gensim-loss-function. To test if a hyperparameter value performs statistically better than another value, an adopted Permutationstest could be done. The
evalute_generated_models.py file gives as output tables for all hyperparameter.

To avoid choosing a model which includes statistically the best hyperparameters but not the best combination, this model is compared with the 5 best performing generated model. The model with statistically the best hyperparameter should perform at least as good as the best of the other model.
 The file print_compare_models.py has as output a graph for this question.

## Finale model
I don't save the hundreds of models I generated. Therefore, the final model must be built (build_final_model.py).
To make sure, the model could be used for gender bias to analyze a validation is need. This could be done with a print of all word token. The gender words must build clusters, if not, the model does not understand gender or has no gender bias at all (validate_model.py).

## WEAT 
Now the gender bias detection could be started. WEAT are the most come testing, but there could be other tests done as well (weat.py).

