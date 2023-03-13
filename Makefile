.PHONY: install
install:
	python setup.py install

.PHONY: test
test:
	python setup.py test

.PHONY: experiments_scheduler
experiments_scheduler:
	@bash ./experiments/ucr_experiments.sh --scheduler

.PHONY: experiments
experiments:
	@bash ./experiments/ucr_experiments.sh

.PHONY: download-ucr-archive
download-ucr-archive:
	wget https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip
	unzip UCRArchive_2018.zip
