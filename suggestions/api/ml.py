import ssl
import simplejson
import sys
sys.path.insert(0, "api")
import tryy
from flask import Flask, flash, request, abort, jsonify
sys.path.insert(0, "sports")
import SearchStringAnalysis

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('api/server.crt', 'api/server.key')

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


# API for the competitor list
@app.route("/", methods=['POST', 'GET'])
def suggestion():
    with open('api/api.key', 'r') as apikey:
        key = apikey.read().replace('\n', '')
        name = request.form['name']
        m=request.form.get('max_l')
        if m:
            max_l=int(request.form['max_l'])
        else:
            max_l=200
        c = request.form.get('count_c')
        if c:
            count_c= request.form['count_c']
        else:
            count_c='c'
        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == key:
            ulos = tryy.similar(name, max_l, count_c)
            return jsonify(ulos)
        else:
            abort(401)


# API for comparison between two company descriptions
@app.route("/sports", methods=['POST', 'GET'])
def sports():
    with open('api/api.key', 'r') as apikey:
        key = apikey.read().replace('\n', '')
        name = request.form['name']

        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == key:
            out = SearchStringAnalysis.parse_and_match(name)
            return_dict=dict()

            return jsonify(out)
        else:
            abort(401)


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=3000)

