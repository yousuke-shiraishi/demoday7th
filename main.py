# -*- coding: UTF-8 -*-
import shutil
import tempfile
import requests
from flask import redirect, url_for, render_template, flash, request, abort
from flask import Flask
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
# import json
from flask import send_from_directory
import numpy as np
import cv2
from datetime import datetime
import string
import glob
from PIL import Image
from botocore.client import Config
# from flask.logging import create_logger
# from flask_login import LoginManager, login_required, login_user, logout_user, current_user
# from flask_wtf import Form
# from wtforms import TextField, PasswordField, validators
# import ldap
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
from dotenv import load_dotenv
import psycopg2
import boto3

# load_dotenv(verbose=True)


# dotenv_path = join(dirname(__file__), '.env')
# load_dotenv(dotenv_path)

app = Flask(__name__)
# LOG = create_logger(app)
# login_manager = LoginManager()
# login_manager.login_view =  "login"
# login_manager.init_app(app)

# app.config['SECRET_KEY'] = "secret"
# app.config['LDAP_URL'] = 'ldap://localhost:389'
# app.config['LDAP_DN_FORMAT'] = 'cn=%s,ou=Users,dc=example,dc=com'

# class User(object):
#     def __init__(self, username, data=None):
#         self.username = username
#         self.data = data
#     def is_authenticated(self):
#         return True
#     def is_active(self):
#         return True
#     def is_anonymous(self):
#         return False
#     def get_id(self):
#         return self.username

#     @classmethod
#     def auth(cls, username, password):
#         l = ldap.initialize(app.config['LDAP_URL'])
#         dn = app.config['LDAP_DN_FORMAT'] % (username)
#         try:
#             l.simple_bind_s(dn, password)
#         except:
#             return None
#         return User(username)

# class LoginForm(Form):
#     username = TextField('Username', validators=[validators.Required()])
#     password = PasswordField('Password', validators=[validators.Required()])

#     def __str__(self):
#         return '''
# <form action="/login" method="POST">
# <p>%s: %s</p>
# <p>%s: %s</p>
# %s
# <p><input type="submit" name="submit" /></p>
# </form>
# ''' % (self.username.label, self.username,
#        self.password.label, self.password,
#        self.csrf_token)

# @login_manager.user_loader
# def load_user(username):
#     return User(username)

# @app.route('/admin')
# @login_required
# def index_user():
#     # カーソル作成
#     conn = psycopg2.connect(
#         host = "127.0.0.1",
#         port = 5432,
#         database="flask-similar",
#         user="postgres",
#         password="postgres")
#     cur = conn.cursor()

#     ins_img_path = request.files['insert_img']
#     ins_img_url = ins_img_path[0]
#     if not ins_img_path.endswith(('.png', '.jpg', '.jpeg')):
#         print('Specify LINE_CHANNEL_SECRET as environment variable.')
#         sys.exit(1)
#     stream = ins_img_path.stream
#     img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
#     img = cv2.imdecode(img_array, 1)
#     img_size = (200, 200)
#     target_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     target_img = cv2.resize(target_img, img_size)

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     # detector = cv2.ORB_create()
#     detector = cv2.AKAZE_create()
#     (_, target_des) = detector.detectAndCompute(target_img, None)
#     # SQLコマンド実行 (プレースホルダー使用。エスケープも勝手にされる)
#     cur.execute("INSERT INTO flask_similar (image_name, image_json_feature) VALUES (%s, %s)", (ins_img_url, json.dumps({ins_img_url:target_des})))
#     # SQL結果を受け取る
#     # コミット
#     conn.commit()
#     # クローズ
#     cur.close()
#     conn.close()

# def index_user():
#     # カーソル作成
#     conn = psycopg2.connect(
#         host = "127.0.0.1",
#         port = 5432,
#         database="flask-similar",
#         user="postgres",
#         password="postgres")
#     cur = conn.cursor()

#     ins_img_path = request.files['insert_img']
#     # SQLコマンド実行 (プレースホルダー使用。エスケープも勝手にされる)
#     cur.execute("INSERT INTO flask_similar (image_name, image_json_feature) VALUES (%s, %s)", (ins_img_url, json.dumps({ins_img_url:target_des})))
#     # SQL結果を受け取る
#     # コミット
#     conn.commit()
#     # クローズ
#     cur.close()
#     conn.close()





# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     form = LoginForm(request.form)
#     if form.validate_on_submit():
#         user = User.auth(form.username.data, form.password.data)
#         if user:
#             login_user(user)
#             print('Login successfully.')
#             return redirect(request.args.get('next', '/'))
#         else:
#             print('Login failed.')
#     return str(form)

# @app.route("/logout")
# def logout():
#     logout_user()
#     return redirect('/login')
###########################################################
# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

DATABASE_URL = os.environ['DATABASE_URL']




SUB_DIR = 'actress/'
app.secret_key = os.getenv('SECRET_KEY', 'for dev')
SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)


# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = SAVE_DIR


AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_STORAGE_BUCKET_NAME = os.environ['AWS_STORAGE_BUCKET_NAME']
AWS_REGION_NAME = os.environ['AWS_REGION_NAME']


# POSTG_ID = os.environ['PG_ID']
# POSTG_PW = os.environ['PG_PW']
# POSTG_DB = os.environ['PG_DB']

estimated_d =[]
img1 =[]
exists_img=[]
img_url=""

# s3 = boto3.client('s3')
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
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    s3 = boto3.client('s3', region='ap-northeast-1',config=Config(signature_version='s3v4'))
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
        print("aaaaa",img_path)
        estimated_d.append(file[1])
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
    s3 = boto3.client('s3', region='ap-northeast-1',config=Config(signature_version='s3v4'))
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    message_content = line_bot_api.get_message_content(event.message.id)
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)

    # 画像として読み込み
    # img1 = content
    # stream = img1.stream
    # img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img_array = np.asarray( BytesIO(message_content.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    img_size = (200, 200)
    ret = {}

#     #####################################
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
#     ############################################################
    
    
    
    dic_sorted = sorted(ret.items(), reverse=True,key=lambda x:x[1])[:3]
    estimated_d =[]
    exists_img =[]
    for file in dic_sorted:
        img_path = s3.generate_presigned_url(
        ClientMethod = 'get_object',
        Params = {'Bucket' : AWS_STORAGE_BUCKET_NAME, 'Key' : "actress/"+ file[0]},
        ExpiresIn = 10,
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
        messages =[
        ImageSendMessage(original_content_url = exists_img[0]),
        TextSendMessage(text=estimated_d[0]),
        ImageSendMessage(original_content_url = exists_img[1]),
        TextSendMessage(text=estimated_d[1]),
        ImageSendMessage(original_content_url = exists_img[2]),
        TextSendMessage(text=estimated_d[2])]
    )
        



if __name__ == '__main__':
    app.debug = True
    app.run()
    # import ssl
    # app.run(host='0.0.0.0', port=5955, ssl_context=('server.crt', 'server.key'), threaded=True, debug=True)
