# Disaster Response Pipeline  
  
For this project, I used disaster-related message data from [Figure Eight](https://www.figure-eight.com) to create the web app to classify disaster text messages into multiple categories. The data contains real messages that were sent during disaster events along with labels categorizing the messages. Messages can have more then one label.   
  
Utilizing Pandas and SQLalchemy, I created an ETL pipeline to load, clean, and save the data into a SQL database. Using Scikit Learn's Pipeline and GridSearchCV modules, I created an ML pipeline to process the messages with NLP and train an Adaboost multi-classification model. Finally, utilizing Plotly and Flask, I created a web app to interface with the trained model to classify new disaster messages and visualize some data exploration. For information about the web app, see the Results below.    
  
  
## Installation  
  
The libraries required for this code are listed in 'requirements.txt'. In order to install all the libraries: `pip3 install -r requirements.txt`.  
  
  
## File Description  
  
- app  
  - template  
    \- go.html: classification result page of web app  
    \- master.html: main page of web app   
  - run.py: Flask file that runs app  
  
- data  
  - disaster_categories.csv: data to process   
  - disaster_messages.csv: data to process  
  - DisasterResponse.db: database to save clean data into  
  - process_data.py: module to run ETL pipeline    
  
- images  
  - disaster-response-project1.png  
  - disaster-response-project2.png  
  
- models  
  - classifier.pkl: saved model  
  - train_classifier.py: module to run ML pipeline  
  
- notebooks  
  - ETL_Pipeline_Preparation.ipynb  
  - ML_Pipeline_Preparation.ipynb     
   
- README.md  
- requirements.txt: text of the python packages rquired to run this project  
  
  
## Instructions  
  
To retrain the model:   
1. Run the ETL pipeline to load, clean, & save the data into the database. In the terminal, run `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`.  
2. Run the ML pipeline to NLP the data, train the classifier, and save a model. In the terminal, run `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`.  
   
To run the web app:  
1. In the terminal, run `python app/run.py`.  
2. Go to http://localhost:3001/.  
   
  
## Results  
  
This project includes a web app to classify new disaster messages. First, when you navigate to http://localhost:3001/, the web app displays from visualizations about the data used to train the classification model.   
![Data Visualizations](https://github.com/janniec/DisasterResponsePipeline/blob/master/images/disaster-response-project1.png)   
  
In addition, the web app has an interface to input new disaster messages for classification.   
![Classification Interface](https://github.com/janniec/DisasterResponsePipeline/blob/master/images/disaster-response-project2.png)   
  
  
## Next step  
  
Add some more visulizations to the web app.  
  
  