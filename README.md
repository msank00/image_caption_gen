# :camera: :bookmark_tabs: Image Captioning

## :bulb: Idea:

- [blog](https://towardsdatascience.com/automatic-image-captioning-with-cnn-rnn-aae3cd442d83)
- [Github](https://github.com/Noob-can-Compile/Automatic-Image-Captioning)
  - [nbviewer](https://nbviewer.jupyter.org/github/Noob-can-Compile/Automatic-Image-Captioning/tree/master/)
- [Create Vocabulary in NLP tasks](https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html)

## :floppy_disk: Dataset

- [dataset](https://www.kaggle.com/shadabhussain/flickr8k)

## :broom: Data Cleaning

```py
python 00_clean_data.py
```

## :chart_with_upwards_trend: Data Validation

For quick data validation run the below line at the terminal from project parent directory

```py
streamlit run 01_check_data.py
```

This will create and open a simple [streamlit](https://www.streamlit.io/) **data-app** in the browser. Set the `slider`  and check differnt `image sample` and corresponding `caption`

![image](asset/demo.png)

## :rocket: Train Model

```py
make train
```

## :rocket: Predict

```py
make predict
```

## :lock: Track experement

- Use [comet ml](https://www.comet.ml/site/)

## :dart: TODO:

1. [x] Create DataSet and DataLoader
   1. [x] Create Image Transform
2. [x] Create Network
   1. [x] Create Encoder
   2. [x] Create Decoder
      1. [x] Check the LSTM syntaxt, pytorch documentation
3. [x] Optimizer
4. [x] Loss/objective function/criterion
5. [x] Add reproducibility
6. [x] Train model
   1. [ ] Callbacks
      1. [x] Learning rate scheduler, ..plateaue
      2. [ ] Saving best model
7. [ ] **Performance Evaluation**
   1. [ ] Add Sentence level **BLEU score** to compare true captions and predicted captions. [link](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
   2. [ ] [METEOR Score](https://www.nltk.org/api/nltk.translate.html) Metric for Evaluation of Translation with Explicit ORdering:  
   3. [ ] [CIDEr](http://vrama91.github.io/cider/) (Consensus-based Image Description Evaluation): Used as a measurement for image caption quality
8. [x] Prediction
9. [ ] **Model Debugging** :fire:
   1. Paper to follow:
      1. [Learning cnn lstm architecture for image caption generation Moses Soh](http://cs224d.stanford.edu/reports/msoh.pdf)
         1. [x] As per the paper, 2 LSTM 
         2. [ ] with droupout (keep probability 0.75) work best for MSCOCO dataset
   2. [Coping with Overfitting Problems of Image Caption](https://dacemirror.sci-hub.tw/proceedings-article/6c77b0141a839ab70bfd7c69ed07c4f8/luo2019.pdf?rand=5f218af6655f8?download=true)
   3. [ ] Debug Overfitting
   4. [ ] Vary Learning Rate (learning rate scheduler)
   5. [ ] Vary batch sampler/data loader
   6. [ ] Vary batch size
   7. [x] Add more LSTM layers in the Decoder 
      1. [x] Try Bi-directional
   8. [ ] Add dropout layer
   9. [ ] Add [word embedding](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76)
   10. [x] Check training `input` + `label` order
10. [x] Experiment tracker
11. [ ] Serving
12. [ ] Docker
13. [ ] Deployment (Heroku)

----

:rocket: :rocket:

-----------