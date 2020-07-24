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

## :lock: Track experement

- Use [comet ml](https://www.comet.ml/site/)

## :dart: TODO:

1. [x] Create DataSet and DataLoader
   1. [x] Create Image Transform
2. [x] Create Network
   1. [x] Create Encoder
   2. [x] Create Decoder
      1. [ ] Check the LSTM syntaxt, pytorch documentation
3. [x] Optimizer
4. [x] Loss/objective function/criterion
5. [x] Train model
   1. [ ] Callbacks
      1. [ ] Learning rate scheduler, ..plateaue
      2. [ ] Saving best model...
6. [ ] Prediction
7. [ ] Serving
8. [ ] Docker
9. [ ] Deployment (Heroku)

-----------