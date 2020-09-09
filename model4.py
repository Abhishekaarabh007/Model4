import sklearn
import simplexml
from flask import request,make_response, Flask
from flask_restful import Resource, Api
import numpy as np
import keras
from keras.models import model_from_json

app = Flask(__name__)
api = Api(app, default_mediatype='application/json')

@api.representation('application/xml')
def xml(data, code, headers=None):
    resp = make_response(simplexml.core.dumps({'response': data}), code)
    resp.headers.extend(headers or {})
    return resp


import pickle as pickle


mlp= pickle.load(open("MLP_model.h5","rb"))

class Predict(Resource):

    def get(self):
        mod4_attribs=[]
        #mod4_attribs.append(request.args.get('mod4_attrib0',type=int))
        mod4_attribs.append(request.args.get('mod4_attrib1',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib2',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib3',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib4',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib5',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib6',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib7',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib8',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib9',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib10',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib11',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib12',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib13',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib14',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib15',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib16',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib17',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib18',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib19',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib20',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib21',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib22',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib23',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib24',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib25',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib26',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib27',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib28',type=float))
        mod4_attribs.append(request.args.get('mod4_attrib29',type=float))


        data = np.array(list(data.values()))
        data = np.reshape(data, newshape=(1, 29), order='C')
        yPred1 = mlp.predict_classes(data)
        output = yPred1[0]
        return jsonify(int(output))
       

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
