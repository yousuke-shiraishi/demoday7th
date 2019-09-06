# -*- coding: UTF-8 -*-
import shutil
import tempfile
import requests
from flask import redirect, url_for, render_template, flash, request, abort
from flask import Flask
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
import json
from flask import send_from_directory
import numpy as np
import cv2
from datetime import datetime
import string
import glob
from PIL import Image
from botocore.client import Config
# from flask.logging import create_logger
from flask_httpauth import HTTPDigestAuth
from io import BytesIO

import os
import sys
from argparse import ArgumentParser

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent,
    TextMessage ,ImageMessage, ImageSendMessage,TextSendMessage
)

from os.path import join, dirname
# from dotenv import load_dotenv
import psycopg2
import boto3

# load_dotenv(verbose=True)
# POSTG_ID = os.environ['PG_ID']
# POSTG_PW = os.environ['PG_PW']
# POSTG_DB = os.environ['PG_DB']

# dotenv_path = join(dirname(__file__), '.env')
# load_dotenv(dotenv_path)

app = Flask(__name__)
# LOG = create_logger(app)
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_STORAGE_BUCKET_NAME = os.environ['AWS_STORAGE_BUCKET_NAME']
AWS_REGION_NAME = os.environ['AWS_REGION_NAME']
USER_ID = os.environ['USER_ID']
USER_PD = os.environ['USER_PD']
app.secret_key = os.environ['SECRET_KEY']
DATABASE_URL = os.environ['DATABASE_URL']
auth = HTTPDigestAuth()

users = {
    USER_ID: USER_PD
}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/admin', methods=['GET','POST'])
@auth.login_required
def admin():
    return render_template("admin.html")


@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

@app.route('/admin_ins', methods=['GET','POST'])
@auth.login_required
def admin_ins():
    # カーソル作成
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    # conn = psycopg2.connect(
    # host = "0.0.0.0",
    # port = 5432,
    # database=POSTG_DB,
    # user=POSTG_ID,
    # password=POSTG_PW)
    
    cur = conn.cursor()
    ##############################################
    
    if 'insert_img1' not in request.files:
        flash('ファイルがありません','failed')
        return redirect(request.url)
    fileToUpload = request.files['insert_img1']
                # ファイルのチェック
    if fileToUpload and allowed_file(fileToUpload.filename):
        fileToUpload
    else:
        flash('画像ファイルを入れてください','failed')
        sys.exit(1)

    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    file_url = str(fileToUpload)
    ins_str = file_url.split(" ")
    ins_img_url = ins_str[1][1:-1]
    i = Image.open(fileToUpload)
    save_path = SAVE_DIR +"/" + ins_img_url
    i.save(save_path)
    filename = os.listdir(SAVE_DIR +"/")
    img_size = (200, 200)
    filename1 = SAVE_DIR +"/" + filename[0]
    
    target_img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.resize(target_img, img_size)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # detector = cv2.ORB_create()
    detector = cv2.AKAZE_create()
    (_, target_des) = detector.detectAndCompute(target_img, None)
    # SQLコマンド実行 (プレースホルダー使用。エスケープも勝手にされる)

    cur.execute("INSERT INTO flask_similar (image_name, image_json_feature) VALUES (%s, %s)", (filename[0], json.dumps({filename[0]:target_des.tolist()})))
    # SQL結果を受け取る
    # コミット
    # カーソル作成
    conn.commit()
    # クローズ
    cur.close()
    conn.close()
    s3 = boto3.resource('s3') #S3オブジェクトを取得

    bucket = s3.Bucket(AWS_STORAGE_BUCKET_NAME)
    print("filename[0]",filename[0])
    bucket.upload_file(filename1, 'actress/' + filename[0])
###################################################

    return render_template("admin.html")
     

@app.route('/admin_del', methods=['GET','POST'])
@auth.login_required
def admin_del():
    if request.method == "GET":
        return """
            <h1>
                ファイルを削除してデータベースを更新
            </h1>
            <form action="/admin_del" method = post>
                <p><input type="text" name="delete_img1"></p>
                <input type = submit value = delete_img>
            </form>
        """
    else:
        # カーソル作成
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        # conn = psycopg2.connect(
        # host = "0.0.0.0",
        # port = 5432,
        # database=POSTG_DB,
        # user=POSTG_ID,
        # password=POSTG_PW)
    
        cur = conn.cursor()
        del_img_url = request.form['delete_img1']
        
        # SQLコマンド実行 (プレースホルダー使用。エスケープも勝手にされる)
        s3_client = boto3.client('s3')
        s3_client.delete_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key='actress/' + del_img_url)
        
        sql = 'DELETE FROM flask_similar WHERE image_name = %s'
        cur.execute(sql, (del_img_url,))
        
        # SQL結果を受け取る
        # コミット
        conn.commit()
        # クローズ
        cur.close()
        conn.close()
        return """
            <h1>
                ファイルを削除してデータベースを更新
            </h1>
            <form action="/admin_del" method = post>
                <p><input type="text" name="delete_img1"></p>
                <input type = submit value = delete_img>
            </form>
        """


###########################################################
channel_secret = os.environ['LINE_CHANNEL_SECRET']
channel_access_token = os.environ['LINE_CHANNEL_ACCESS_TOKEN']

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)


SUB_DIR = 'actress/'

SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)


# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = SAVE_DIR



estimated_d =[]
img1 =[]
exists_img=[]
img_url=""




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

###########

@app.route('/upload', methods=['GET','POST'])
def upload():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    # conn = psycopg2.connect(
    # host = "0.0.0.0",
    # port = 5432,
    # database=POSTG_DB,
    # user=POSTG_ID,
    # password=POSTG_PW)
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    s3 = boto3.client('s3', region_name='ap-northeast-1',config=Config(signature_version='s3v4'))
    # # ファイルがなかった場合の処理
    if 'image' not in request.files:
        flash('ファイルがありません','failed')
        return redirect(request.url)
    img1 = request.files['image']
                # ファイルのチェック
    if img1 and allowed_file(img1.filename):
        img1
    else:
        flash('画像ファイルを入れてください','failed')
        sys.exit(1)
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
        # conn = psycopg2.connect(
        # host = "0.0.0.0",
        # port = 5432,
        # database=POSTG_DB,
        # user=POSTG_ID,
        # password=POSTG_PW)
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
        ExpiresIn = 60,
        HttpMethod = 'GET')
        
        if file[1] >= 0.85:
            estimated_d.append("類似度 高")
        elif file[1] >= 0.8:
            estimated_d.append("類似度 中")
        else:
            estimated_d.append("類似度 低")
        exists_img.append(img_path)
        
    return render_template('index.html',img_url=img_url, data= zip(exists_img,estimated_d))



@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    # LOG.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    s3 = boto3.client('s3', region_name='ap-northeast-1',config=Config(signature_version='s3v4'))
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    message_content = line_bot_api.get_message_content(event.message.id)
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    i = Image.open(BytesIO(message_content.content))
    save_path = SAVE_DIR +"/" + event.message.id + '.jpg'
    i.save(save_path)
    filename = os.listdir(SAVE_DIR +"/")
    img_size = (200, 200)
    ret = {}

#     #####################################
    if SUB_DIR == 'actress/':

        filename = SAVE_DIR +"/" + filename[0]
        target_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, img_size)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # detector = cv2.ORB_create()
        detector = cv2.AKAZE_create()
        (_, target_des) = detector.detectAndCompute(target_img, None)
        # conn = psycopg2.connect(
        # host = "0.0.0.0",
        # port = 5432,
        # database=POSTG_DB,
        # user=POSTG_ID,
        # password=POSTG_PW)
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
#     ############################################################
    
    
    
    dic_sorted = sorted(ret.items(), reverse=True,key=lambda x:x[1])[:3]
    estimated_d =[]
    exists_img =[]
    for file in dic_sorted:
        img_path = s3.generate_presigned_url(
        ClientMethod = 'get_object',
        Params = {'Bucket' : AWS_STORAGE_BUCKET_NAME, 'Key' : "actress/"+ file[0]},
        ExpiresIn = 60,
        HttpMethod = 'GET')
        if file[1] >= 0.85:
            estimated_d.append("類似度 高")
        elif file[1] >= 0.8:
            estimated_d.append("類似度 中")
        else:
            estimated_d.append("類似度 低")

        exists_img.append(img_path)
    
    line_bot_api.reply_message(
        event.reply_token,
        [
        ImageSendMessage(original_content_url = exists_img[0],
        preview_image_url=exists_img[0]),
        TextSendMessage(text=estimated_d[0]),
        ImageSendMessage(original_content_url = exists_img[1],
        preview_image_url=exists_img[1]),
        TextSendMessage(text=estimated_d[1])]
    )
        



if __name__ == '__main__':
    app.debug = True
    app.run()
    # import ssl
    # app.run(host='0.0.0.0', port=5955, ssl_context=('server.crt', 'server.key'), threaded=True, debug=True)
