# NBA-Target

Web API containing a Machine Learning model that predicts if a NBA player will stay in the league 5 years later.

![logo](static/img/logo.png)

This model in contained in the `ml_model.py` file and use the `nba_logreg.csv` file that contains the dataset.


## Web page

Web site adapted to a random common user without kwnoledge about REST requests.

Execute the `app_flask.py` file to start the webservice.
Then you can go to the url : [http://localhost:8888/](http://localhost:8888/)


## Simple REST API

Webservice in REST format.

Execute the `api_flask.py` file to start the webservice.
Then you can go to the url : [http://localhost:8888/](http://localhost:8888/)


## Python libraries

Numpy, time, Pandas, seaborn, matplotlib, Scikit-Learn, Flask, Flask-RESTPlus
