# WordPredictorPLT
## An NLP next-word predictor
Frontend created with React and Bootstrap.

Backend hosted locally via Python Flask app (see [predict.py](./predict.py)) that runs a pretrained model. Training done manually using [train.py](./train.py) - given corpus input, this file extracts a vocabulary of words and finds a list of trigrams. The trigrams are then split into an input (first two words) and output (last word). Input is then encoded into a 512-dimensional vector space with Universal Sentence Encoder and trained with a custom model. Output of the model is of size len(vocabulary), and each given input following is treated as a classification problem.

**A video demo of this project can be found [here](https://www.youtube.com/watch?v=NHd5NQ1ePaI).**

Experimental, more lightweight model is being developed in [train_lite.py](./train_lite.py) to test a form of vector mapping from 512-dimensional input to 512-dimensional output, followed by nearest-neighbour identification from vocabulary.

[![](./readme_assets/hosted-site.jpg)](https://www.youtube.com/watch?v=NHd5NQ1ePaI)
