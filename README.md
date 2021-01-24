# Disaster Response Pipeline
During a natural disaster, emergency and response teams are flooded with thousands of messages from different sources (twitter, direct messages etc.) that they are unable to manually inspect. In this project we will use NLP skills to aid them to correctly classify these messages into 36 categories.

### Data and requirements
We will use two CSV files with over 26000 messages already processed by [Figure Eight](https://appen.com/): 
- data/disaster_messages.csv
- data/disaster_categories.csv
- The requirements you will need are listed in `requirements.txt`.

### Overview of the program and basic usage:
 1. ETL Pipeline: ``data/process_data.py``will automatically load, merge and clean the csv files and store them in a SQLite database. Usage:
- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. ML Pipeline: `models/train_classifier.py`will automatically train and save a classifier using an exhaustive search to find the best set of parameters for the model. Usage:
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Web app: I have also provided web app so that an emergency worker can input a new message and get classification results in several categories. In order to run the Flask app:
- Run `python run.py`.
- Go to http://0.0.0.0:3001/




