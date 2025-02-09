#!/usr/bin/python
#coding:utf8

from flask import Flask, render_template, url_for, request, redirect, make_response, session, jsonify, json, send_from_directory
import os, time
from werkzeug.utils import secure_filename
import sys  
import traceback
import json
from locket import img_loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__, static_url_path='', static_folder='')
app.debug = True
app.debug = False
app.config['UPLOAD_FOLDER'] = 'data/'
app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))

img_loader = img_loader.ImgLoader(
    '../models/model.pb', 
    '../models/SavedModelLight_sketch', 
    '../models/SavedModelLight_0000', 
    '../models/SavedModelLight_simpson')

def handleException(e):
    print ('str(e):\t\t', str(e))
    print ('repr(e):\t', repr(e))
    print ('e.message:\t', e.message)
    print ('traceback.print_exc():', traceback.print_exc())
    print ('traceback.format_exc():\n%s' % traceback.format_exc())

@app.route('/upload', methods=['GET', 'POST'])
def step():
    if request.method == 'POST':
        try:
            text = request.get_data(as_text=True)
            data = json.loads(text)
            original_path = 'data/%d.png'%data['timestamp']
            matting_path = 'data/%d_matting.png'%data['timestamp']
            transfer_path = 'data/%d_transfer_%s.png'%(data['timestamp'], data['trans_type'])
            
            try:
                # read image (save the image if not existed)
                img_loader.save_image(data['image'], original_path)
                # Do human matting
                img_loader.apply_matting(original_path, matting_path)
                # Style Transfer
                img_loader.transfer(matting_path, transfer_path, data['trans_type'])

                return make_response(jsonify({
                        'status': 'ok',
                        'data': {
                            'original': original_path,
                            'matting': matting_path,
                            'transfer': transfer_path
                        }
                    }))
            except Exception as e:
                handleException(e)

        except Exception as e:
            handleException(e)
            
    return make_response('Error occurs')


if __name__ == '__main__':
    os.makedirs('data/', exist_ok=True)

    ip = '0.0.0.0'
    app.run(host=ip, port=80)
    # app.run(port=9090)

