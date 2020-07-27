project-template:
	mkdir -p data
	mkdir -p model
	mkdir -p output
	mkdir /root/.kaggle
	cp kaggle.json /root/.kaggle/
	chmod 600 /root/.kaggle/kaggle.json

env:
	pip3 install -r requirements.txt
	python3 -c "import nltk; nltk.download('punkt')"
	apt-get install htop

git-config:
	git config user.email "sankarshan7@gmail.com"
	git config user.name "Sankarshan Mridha"

quick-setup: project-template env git-config

clean-log:
	rm **/*.log

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean: clean-pyc
	rm asset/test_image/*.jpg
	rm -rf asset/test_image/.ipynb_checkpoints

format:
	isort -rc -y .
	black -l 79 .


dataval:
	streamlit run 01_check_data.py

train:
	python3 02_train.py

predict:
	python3 03_prediction.py

clean-data:
	python3 00_clean_data.py

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
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt data/main_caption_data/
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt data/main_caption_data/
	cp data/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt data/main_caption_data/
	mv data/flickr_data/Flickr_Data/Images/* data/images/

prepare_model_dir:
	# export KAGGLE_DATASET_ID="sankarshan7/msank-tweet-sentiment"

	# kaggle datasets init -p output


	# download previous models to keep the output directory in sync with kaggle
	# else old files may get deleted in remote while pushing
	kaggle datasets download sankarshan7/image-caption
	mv *.zip model/
	unzip 'model/*.zip'
	rm model/*.zip
	mv *.pkl model/
	mv *.png model/
	mv sample_new.txt model/
	cp dataset-metadata.json model/

	python3 update_meta_json.py
	touch model/sample_new.txt

publish_output:
	kaggle datasets version -p model -m "Updated data"
