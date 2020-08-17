project-template:
	mkdir -p data
	mkdir -p model
	mkdir -p output
	
env:
	pip3 install -r requirements.txt
	python3 -c "import nltk; nltk.download('punkt')"
	apt-get install htop

kaggle-api:
	mkdir -p /root/.kaggle
	python3 98_prepare_kaggle_key.py
	mv kaggle.json /root/.kaggle/
	chmod 600 /root/.kaggle/kaggle.json


git-config:
	git config user.email "sankarshan7@gmail.com"
	git config user.name "Sankarshan Mridha"



clean-log:
	rm **/*.log

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean: clean-pyc
	rm -rf asset/test_image/.ipynb_checkpoints
	rm asset/test_image/*.jpg
	rm -rf .mypy_cache
	rm -rf .ipynb_checkpoints
	rm -rf src/.ipynb_checkpoints

clean-test-image:
	rm asset/test_image/*.jpg

format:
	isort -rc -y .
	black -l 79 .


dataval:
	streamlit run 01_check_data.py

clean-data:
	python3 00_clean_data.py

train:
	python3 02_train.py

predict:
	python3 03_prediction.py

prediction-check:
	streamlit run 04_check_prediction.py

gpu-available:
	# memory footprint support libraries/code
	python3 99_available_gpu.py

quick-push:
	git add .
	git commit -m "Quick code push from colab"
	git push

data-download:

	pip3 install kaggle

	kaggle datasets download -d shadabhussain/flickr8k
	unzip '*.zip'
	mv model_weights.h5 data/
	mv train_encoded_images.p data/
	mv flickr_data data/
	rm -rf Flickr_Data
	rm *.zip

prep-main-data:
	mkdir -p data/main_caption_data
	mkdir -p data/images
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt data/main_caption_data/
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt data/main_caption_data/
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt data/main_caption_data/
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt data/main_caption_data/
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt data/main_caption_data/
	mv data/flickr_data/Flickr_Data/Images/* data/images/

prepare-model-dir:
	kaggle datasets download sankarshan7/image-caption
	mv *.zip model/
	unzip 'model/*.zip'
	rm model/*.zip
	mv *.png model/
	mv *.pt model/
	python3 96_prepare_dataset_metadata.py
	mv dataset-metadata.json model/
	mv prediction*.csv model/
	mv vocab.pkl model/
	python3 97_update_meta_json.py



publish-output:
	kaggle datasets version -p model -m "Updated data"


quick-setup: project-template kaggle-api env git-config

set-data: data-download prep-main-data prepare-model-dir

pipeline: clean-data train predict publish-output
