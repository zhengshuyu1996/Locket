#!/usr/bin/python
#coding:utf8

from flask import Flask, render_template, url_for, request, redirect, make_response, session, jsonify, json, send_from_directory
import os, time
from werkzeug import secure_filename
import sys  
import traceback
import json


app = Flask(__name__, static_url_path='', static_folder='')
app.debug = False
app.config['UPLOAD_FOLDER'] = 'data/'
app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))

@app.route('/step', methods=['GET', 'POST'])
def step():
    if request.method == 'POST':
        try:
            text = request.get_data(as_text=True)
            with open('test.json', 'w') as f:
                f.write(text)
            outfile = os.popen('./main test.json')
            result = outfile.read()
            result = json.loads(result)
            print(result)
            return make_response(jsonify({"data": result}))
        except Exception as e:
            # print ('str(Exception):\t', str(Exception))
            print ('str(e):\t\t', str(e))
            print ('repr(e):\t', repr(e))
            print ('e.message:\t', e.message)
            print ('traceback.print_exc():', traceback.print_exc())
            print ('traceback.format_exc():\n%s' % traceback.format_exc())
    return send_from_directory(app.config['UPLOAD_FOLDER'],"cards.json")

if __name__ == '__main__':
    app.run(host="35.189.165.204", port=80)
