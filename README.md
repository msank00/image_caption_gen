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

### Good prediction

```py
# Good
# 3375070563_3c290a7991.jpg
# 3610683688_bbe6d725ed.jpg
# 3316725440_9ccd9b5417.jpg
# 3262075846_5695021d84.jpg
# 2901880865_3fd7b66a45.jpg

# moderate
# 542179694_e170e9e465.jpg,
# 2893374123_087f98d58a.jpg
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
      2. [x] Saving best model
7. [ ] **Performance Evaluation**
   1. [ ] Add Sentence level **BLEU score** to compare true captions and predicted captions. [link](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
   2. [ ] [METEOR Score](https://www.nltk.org/api/nltk.translate.html) Metric for Evaluation of Translation with Explicit ORdering:  
   3. [ ] [CIDEr](http://vrama91.github.io/cider/) (Consensus-based Image Description Evaluation): Used as a measurement for image caption quality
8. [x] Prediction
9. [ ] **Model Debugging** :fire:
   1. Paper to follow:
      1. [Learning cnn lstm architecture for image caption generation Moses Soh](http://cs224d.stanford.edu/reports/msoh.pdf)
         1. [x] As per the paper, 2 LSTM 
         2. [x] with [droupout](https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/) (keep probability 0.75) work best for MSCOCO dataset
   2. [Coping with Overfitting Problems of Image Caption](https://dacemirror.sci-hub.tw/proceedings-article/6c77b0141a839ab70bfd7c69ed07c4f8/luo2019.pdf?rand=5f218af6655f8?download=true)
   3. [x] Debug Overfitting
   4. [x] :rocket: **Debug Decoder:** It seems the main issue is the decoder. This [blog](https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3) helped a lot to understand the nuances properly. And finally meaningful captions started to generate.
   5. [x] Vary Learning Rate ([pytorch learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate))
   6. [ ] When to use `softmax()` and relation with loss function
   7. [ ] Vary batch sampler/data loader
   8. [ ] Vary batch size
   9. [x] Add more LSTM layers in the Decoder 
      1. [x] Try Bi-directional
   10. [x] Add dropout layer
   11. [ ] Add [word embedding](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76)
   12. [x] Check training `input` + `label` order
10. [x] Experiment tracker
11. [ ] Serving
12. [ ] Docker
13. [ ] Deployment (Heroku)

## Learning

The decodre part is tricky. Initially I was using the `nn.LSTM()` which actually trains in bulk, i.e small lstm cells [blue boxes in the below image] are already packed based on cofiguration [refer below image]. This was causing issues while doing prediction. Somehow, I was missing the connection of how does it make sure that `hidden_sate` and `cell_sate` 
at time `t-1` are fed at next time step `t`, i.e, following the definition of the traditional `LSTM`. May be it can be done using the `nn.LSTM()` module. But I was unable to do it. And due to this, during the initial training days, the output captions were not making senses. 

**LSTM Implementation in PyTorch**

![image](https://i.stack.imgur.com/SjnTl.png)

After going through this [blog](https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3) it's understood that basics of LSTM should be used for validation of understanding the concept and generate word one at a time in a loop following the principle of LSTM definition. So in the update Decoder, `nn.LSTMCell()` [blue boxes] is used, which is the building block of `nn.LSTM()` module. And captions are generated using `nn.LSTMCell()` in a loop and finally predicted captions start to make sense. Using `nn.LSTMCell()` is like using one blue box at a time for each time steps. 

Great learning !!

### How to interpret the pytorch LSTM module?

It really depends on a model you use and how you will interpret the model. Output may be:

- a single LSTM cell hidden state
- several LSTM cell hidden states
- all the hidden states outputs

Output, is _almost never interpreted directly_. If the input is encoded there should be a softmax layer to decode the results.

Note: In language modeling hidden states are used to define the probability of the next word, p(wt+1|w1,...,wt) =softmax(Wht+b).


**Reference:**

- [What's the difference between “hidden” and “output” in PyTorch LSTM?](https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm) :fire:
 

----

:rocket: :rocket:

-----------