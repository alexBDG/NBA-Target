# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:04:26 2020

@author: Alexandre Banon
"""

from flask import Flask, request, url_for, redirect, session
from flask_restplus import Api, Resource, fields

from ml_model import newModel

import numpy as np


app = Flask(__name__)
api = Api(app=app, version='0.1', title='NBA Players Api', description='', validate=True)
app.secret_key = "azerty"

ns_players = api.namespace('players', description = "players operations")

player_definition = api.model("Player Informations", {
    "Name": fields.String(required=True, description="Player Name"),
    "GP": fields.Float(required=True, description="Matchs joués"),
    "MIN": fields.Float(required=True, description="Minutes"),
    "PTS": fields.Float(required=True, description="Points"),
    "FG%": fields.Float(required=True, description="Pourcentage de tir réussis"),
    "3P Made": fields.Float(required=True, description="Tirs à 3 points réussis"),
    "3P%": fields.Float(required=True, description="Pourcentage de tir à 3 points réussis"),
    "FTM": fields.Float(required=True, description="Lancers francs réussis"),
    "FT%": fields.Float(required=True, description="Pourcentage de lancers francs réussis"),
    "OREB": fields.Float(required=True, description="Rebonds offensifs"),
    "DREB": fields.Float(required=True, description="Rebonds défensifs"),
    "AST": fields.Float(required=True, description="Passe décisive"),
    "STL": fields.Float(required=True, description="Interceptions"),
    "BLK": fields.Float(required=True, description="Contres"),
    "TOV": fields.Float(required=True, description="Ballons perdus")
})


list_players = api.model('Player', {
    'players': fields.List(fields.Nested(player_definition))
})


# Initialize the model
mod = newModel()

# Train the model
(score, train_time) = mod.trainModel()

          
def infoSession():
    X_player = []
    for el in mod.paramset:
        X_player += [float(session[el])]
    return np.array([X_player])


@app.route('/prediction/')
def prediction():
    X_test = infoSession()
    (pred, pred_time) = mod.predictModel(X_test)
    if pred>0.5:
        invest = "Oui"
        message = "devrait"
    else:
        invest = "Non"
        message = "ne devrait pas"
    return "Investir sur {0} ?\n{1}\nD'après l'annalyse de ses performances, il {2} rester en NBA d'ici 5 ans.".format(session["Name"], invest, message)


@api.route("/player/")
class playersList(Resource):
    @api.expect(player_definition)
    @api.response(200, 'NBA Target Prediction: Success')
    @api.response(400, 'NBA Target Prediction: Validation error')
    def post(self):
        """
        Add a new player in order to make a prediction.
        """
        data = request.get_json()
        for el in list(data):
            session[el] = data[el]
        return redirect(url_for('prediction'))
           
        

app.run(port= 8888, host= "localhost")