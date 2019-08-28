# -*- coding: UTF-8 -*-
import shutil
import tempfile
import requests
from flask import redirect, url_for, render_template, flash, Flask, request, abort
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
from flask import send_from_directory
import numpy as np
import cv2
from datetime import datetime
import string
import glob
from PIL import Image
import pickle
    

import os
import sys
from argparse import ArgumentParser

# from linebot import (
#     LineBotApi, WebhookHandler
# )
# from linebot.exceptions import (
#     InvalidSignatureError
# )
# from linebot.models import (
#     MessageEvent,
#     TextMessage ,ImageMessage
# )

from os.path import join, dirname
from dotenv import load_dotenv
import psycopg2
import boto3
from boto3.session import Session

load_dotenv(verbose=True)


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)

# # get channel_secret and channel_access_token from your environment variable
# channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
# channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
# if channel_secret is None:
#     print('Specify LINE_CHANNEL_SECRET as environment variable.')
#     sys.exit(1)
# if channel_access_token is None:
#     print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
#     sys.exit(1)

# line_bot_api = LineBotApi(channel_access_token)
# handler = WebhookHandler(channel_secret)

# DATABASE_URL = os.environ['DATABASE_URL']

# conn = psycopg2.connect(DATABASE_URL, sslmode='require')



SUB_DIR = 'actress/'
app.secret_key = os.getenv('SECRET_KEY', 'for dev')
SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

IMG_DIR = './images_stock/'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = SAVE_DIR


AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_STORAGE_BUCKET_NAME = os.environ['AWS_STORAGE_BUCKET_NAME']
AWS_REGION_NAME = os.environ['AWS_REGION_NAME']


POSTG_ID = os.environ['PG_ID']
POSTG_PW = os.environ['PG_PW']
POSTG_DB = os.environ['PG_DB']

estimated_d =[]
img1 =[]
exists_img=[]
img_url=""

s3 = boto3.client('s3')
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET','POST'])
def index():
    
    if exists_img ==[]:
        estimated_d=[]
        return render_template("index.html")
    else:
        return render_template("index.html",img_url=img_url, data= zip(exists_img,estimated_d))



@app.route('/upload', methods=['GET','POST'])
def upload():
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)

        

    # # ファイルがなかった場合の処理
    # if 'file' not in request.files:
    #     flash('ファイルがありません','failed')
    #     return redirect(request.url)
    # file = request.files['image']
    #             # ファイルのチェック
    # if file and allowed_file(file.filename):
        # 危険な文字を削除（サニタイズ処理）
        # filename = secure_filename(file.filename)
    # 画像として読み込み
    img1 = request.files['image']
    stream = img1.stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    img_size = (200, 200)
    ret = {}
    Img =  Image.open(img1)
    dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_%f")
    save_path = os.path.join(SAVE_DIR, dt_now + ".jpeg")
    Img.save(save_path)

    img1 = glob.glob(save_path)
    img_url = img1[0]
    
    #####################################3

    if SUB_DIR == 'actress/':

    
        target_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        target_img = cv2.resize(target_img, img_size)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # detector = cv2.ORB_create()
        detector = cv2.AKAZE_create()
        (_, target_des) = detector.detectAndCompute(target_img, None)
        conn = psycopg2.connect(
        host = "0.0.0.0",
        port = 5432,
        database=POSTG_DB,
        user=POSTG_ID,
        password=POSTG_PW)
        c = conn.cursor()
        c.execute('SELECT * FROM flask_similar')
        rows = c.fetchall()
        for row in rows:
            if not row[1].endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                numpy_img_data = np.array(row[2][row[1]]).astype(np.uint8)
                matches = bf.match(target_des, numpy_img_data)
                dist = [m.distance for m in matches]
                score = sum(dist) / len(dist)
                if score <= 100:
                    score = 100
                score = 100.0 / score
            except cv2.error:
                score = 100000
            ret[row[1]] = score
        conn.close()

    ############################################################
    
    dic_sorted = sorted(ret.items(), reverse=True,key=lambda x:x[1])[:3]
    estimated_d =[]
    exists_img =[]
    for file in dic_sorted:
        img_path = s3.generate_presigned_url(
        ClientMethod = 'get_object',
        Params = {'Bucket' : AWS_STORAGE_BUCKET_NAME, 'Key' : "actress/"+ file[0]},
        ExpiresIn = 10,
        HttpMethod = 'GET')
        # img_path = get_public_url(AWS_STORAGE_BUCKET_NAME,"actress/"+ file[0])
        # obj = s3.Object(AWS_STORAGE_BUCKET_NAME, "actress/"+ file[0])
        print(img_path)
        estimated_d.append(file[1])
        exists_img.append(img_path)
        
    return render_template('index.html',img_url=img_url, data= zip(exists_img,estimated_d))



# @app.route("/callback", methods=['POST'])
# def callback():
#     # get X-Line-Signature header value
#     signature = request.headers['X-Line-Signature']

#     # get request body as text
#     body = request.get_data(as_text=True)
#     app.logger.info("Request body: " + body)

#     # handle webhook body
#     try:
#         handler.handle(body, signature)
#     except InvalidSignatureError:
#         abort(400)

#     return 'OK'

# @handler.add(MessageEvent, message=(ImageMessage))
# def handle_image_message(event):
#     content = line_bot_api.get_message_content(event.message.id)
#     shutil.rmtree(SAVE_DIR)
#     os.mkdir(SAVE_DIR)

#     # 画像として読み込み
#     # img1 = content
#     # stream = img1.stream
#     stream = content
#     img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
#     img = cv2.imdecode(img_array, 1)
#     img_size = (200, 200)
#     ret = {}
#     com_img_dists = {}
    
#     # Img =  Image.open(img1)
#     Img =  Image.open(stream)
#     dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_%f")
#     save_path = os.path.join(SAVE_DIR, dt_now + ".jpeg")

    
#     Img.save(save_path)

#     img1 = glob.glob(save_path)
#     img_url = img1[0]

#     img_dir_files = cloudinary.api.resources(
#     type = "upload", 
#     prefix = "actress")
#     f1 = open('file_name_texts','wb')
#     pickle.dump(img_dir_files,f1)
#     f1.close

#     # f = open('file_name_texts','rb')
#     # img_dir_files = pickle.load(f)

    
    
    
    
    
#     #####################################

    
#     target_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     target_img = cv2.resize(target_img, img_size)

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     # detector = cv2.ORB_create()
#     detector = cv2.AKAZE_create()
#     (_, target_des) = detector.detectAndCompute(target_img, None)

#     comparing_files = img_dir_files
#     # f = open('sample.binaryfile','rb')
#     # com_img_dists = pickle.load(f)
#     for comparing_file in comparing_files:
#         if comparing_file == '.DS_Store':
#             continue
#         if not comparing_file.endswith(('.png', '.jpg', '.jpeg')):
#             continue

#         comparing_img_path = comparing_file
#         try:
#             comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
#             comparing_img = cv2.resize(comparing_img, img_size)
#             (_, comparing_des) = detector.detectAndCompute(comparing_img, None)
#             matches = bf.match(target_des, comparing_des)
#             # matches = bf.match(target_des, com_img_dists[comparing_file])
#             dist = [m.distance for m in matches]
#             score = sum(dist) / len(dist)
#             if score <= 100:
#                 score = 100
#             score = 100.0 / score
#         except cv2.error:
#             score = 100000
#         com_img_dists[comparing_file] = comparing_des
#         ret[comparing_file] = score
#     f = open('sample.binaryfile','wb')
#     pickle.dump(com_img_dists,f)
#     f.close

#     ############################################################
    
    
    
#     dic_sorted = sorted(ret.items(), reverse=True,key=lambda x:x[1])[:3]
#     estimated_d=[]
#     for file in dic_sorted:
#         img_path = cloudinary.CloudinaryImage('actress/' + file[0])
#         img = cv2.imread(img_path)
#         # cv2.imshow('image',img)
#                 # 保存
#         dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_%f")
#         save_path = os.path.join(SAVE_DIR, dt_now + ".jpeg")
#         cv2.imwrite(save_path, img)
#         estimated_d.append(file[1])
#     f_imgs = os.listdir(SAVE_DIR)
#     if '.DS_Store' in f_imgs:
#         f_imgs.remove('.DS_Store')
#     exists_img = sorted(f_imgs)[-3:]
        

    

#     # ファイルの保存
#     # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#     # アップロード後のページに転送
#     return render_template('index.html',img_url=img_url, data= zip(exists_img,estimated_d))


if __name__ == '__main__':
    # app.debug = True
    # app.run()
    import ssl
    app.run(host='0.0.0.0', port=5955, ssl_context=('server.crt', 'server.key'), threaded=True, debug=True)

    



    





