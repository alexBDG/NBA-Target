from flask import Flask, render_template, request, url_for, redirect, session
from ml_model import newModel
import numpy as np

def create_app():
    app = Flask(__name__)
    app.secret_key = "azerty"
    
    # Initialize the model
    mod = newModel()
    
    # Train the model
    (score, train_time) = mod.trainModel()
    
    # Rentrer les explications de chaque attribut s'il est présent dans la base de données
    explanation = {"GP": "Matchs joués",
                   "MIN": "Minutes",
                   "PTS": "Points",
                   "FG%": "Pourcentage de tir réussis",
                   "3P Made": "Tirs à 3 points réussis",
                   "3P%": "Pourcentage de tir à 3 points réussis",
                   "FTM": "Lancers francs réussis",
                   "FT%": "Pourcentage de lancers francs réussis",
                   "OREB": "Rebonds offensifs",
                   "DREB": "Rebonds défensifs",
                   "AST": "Passe décisive",
                   "STL": "Interceptions",
                   "BLK": "Contres",
                   "TOV": "Ballons perdus"}
    paramset = []
    for el in mod.paramset:
        if el in explanation:
            paramset += [[el,explanation[el]]]
        else:
            paramset += [[el,""]]
            
    def infoSession():
        X_player = []
        for el in mod.paramset:
            X_player += [float(session[el])]
        return np.array([X_player])
            

    @app.route('/')
    def homepage():
        return render_template('homepage.html')

    @app.route('/about/')
    def about():
        return render_template('about.html')

    @app.route('/model/')
    def model():
        return render_template('model.html',
                               score=round(score,2),
                               train_time=round(train_time,3))

    @app.route('/prediction/')
    def prediction():
        X_player = infoSession()
        (pred, pred_time) = mod.predictModel(X_player)
        if pred>0.5:
            invest = "Oui"
        else:
            invest = "Non"
        return render_template('prediction.html',
                               name=session["name"],
                               invest=invest,
                               pred=round(pred[0],2))

    @app.route('/recherche/',methods = ['POST', 'GET']) 
    def recherche():
        if request.method == 'POST':
            session["name"] = request.form['name']
            for param in paramset:
                # Permet de récupérer le premier élément car en html, ce qui est après l'espace est supprimé
                _param = param[0].strip().split()[0]
                session[param[0]] = request.form[_param]
            return redirect(url_for('prediction'))
        else:
            return render_template('recherche.html',
                                   paramset=paramset)


    return app




if __name__ == '__main__':
    
    app = create_app()
    app.run(port= 8888, host= "localhost")