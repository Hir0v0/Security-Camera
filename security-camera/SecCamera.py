# -*- coding: utf-8 -*-

from picamera import PiCamera
from time import sleep
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from concurrent.futures import ThreadPoolExecutor # スレッド処理
import RPi.GPIO as GPIO
import datetime as dt
import sys
sys.path.append('/home/pi/.local/lib/python3.7/site-packages') # Pathを明示的に指定
import os
import io
import re
import json
import requests # LINEメッセージ
import subprocess # MP4Boxコマンド実行の為
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter # TesorFlow

SC_CAMERA_PIN = 5 # ピンの名前を変数として定義
MAX_REC_TIME = 100000 # 最長録画時間（秒数）
SAVE_DIR = "./video/" # ファイル保存用ディレクトリ
INITIAL_FILE= "./cert/initial.json" # 初期設定ファイル
LINE_URL = "https://notify-api.line.me/api/notify" # LINEメッセージ用URL
DRIVE_LINK = "https://drive.google.com/open?id=" # LINEに表示するGoogleDriveへのリンク
INTERVAL = 0.2 # 監視間隔（秒）
AN_TEXT_SIZE = 24 # 録画画像上部に表示される注釈文字のサイズ
LABEL_FILE = './models/coco_labels.txt'
MODEL_FILE = './models/mobilenet_ssd_v2_coco_quant_postprocess.tflite'
THRESHOLD = 0.4

GPIO.setmode(GPIO.BCM) # ピンをGPIOの番号で指定
GPIO.setup(SC_CAMERA_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # GPIOセットアップ

class  SecurityCameraClass: # セキュリティカメラのクラス
    def __init__(self):
        with open(INITIAL_FILE) as f: # 初期設定ファイルの読み込み
            __jsn = json.load(f)
            self.folder_id = __jsn["folder_id"] # folder_idの読み込み
            self.token = __jsn["line_token"] # LINE用tokenの読み込み
            self.location = __jsn["location"] # 監視カメラ設置場所
        self.camera = PiCamera()
        self.camera.rotation = 0 # カメラ回転
        gauth = GoogleAuth() # GoogleDriveへの認証
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)

    def start_recordings(self): # 録画開始
        __time_stamp_file =  "{0:%Y-%m-%d-%H-%M-%S}".format(dt.datetime.now()) # 日付時刻をセット
        __time_stamp_disp =  "{0:%Y/%m/%d %H:%M:%S}".format(dt.datetime.now()) # 日付時刻をセット（表示用）
        self.save_file_path = SAVE_DIR + "video" + __time_stamp_file + ".h264" # ディレクトリ、ファイル名をセット
        self.camera.annotate_text = self.location + " " + __time_stamp_disp # 映像上部に表示 
        self.camera.annotate_text_size = AN_TEXT_SIZE
        self.camera.start_recording(self.save_file_path) # 指定pathに録画
        return True 

    def stop_recordings(self, objs): # 録画終了
        self.camera.stop_recording() # 録画終了
        __executor = ThreadPoolExecutor(max_workers=5) # 同時実行は5つまでスレッド実行
        __executor.submit(self.on_theread(self.save_file_path, objs))
        return False 

    def on_theread(self, sv_file, objs): # ファイルのGoogleDriveへのアップロードは時間がかかるので別スレッドで行う
        if len(objs) == 0:
            os.remove(sv_file) # 検出Objectがゼロならファイルを削除して終了
            pass
        __mp4_file_path = os.path.splitext(sv_file)[0] + '.mp4' # 拡張子をmp4にしたファイル名
        __file_name = os.path.basename(__mp4_file_path) # ファイル名部分を取り出し
        # h264形式からmp4に変換
        __res = subprocess.call("MP4Box -add " + sv_file + " " + __mp4_file_path, shell=True)
        if __res == 0: # 変換が正常終了ならファイルをアップロード
            __f = self.drive.CreateFile({"title": __file_name, # GoogleDrive 
                                  "mimeType": "video/mp4",
                                  "parents": [{"kind": "drive#fileLink", "id":self.folder_id}]})
            __f.SetContentFile(__mp4_file_path) # ファイル名指定
            __f.Upload() # アップロード
            os.remove(sv_file) # アップロード後にファイルは削除
            os.remove(__mp4_file_path) # アップロード後にファイルは削除
            sec_camara.line_message(objs) # LINEにメッセージを送信

    def line_message(self, objs): # LINEに録画検知しましたメッセージを送信する
        __headers = {"Authorization" : "Bearer " + self.token}
        __message = objs[0] + " を検知しました " + DRIVE_LINK + self.folder_id
        payload = {"message" : __message}
        __files = {'imageFile': open(self.img_file_path, "rb")} # 画像ファイル
        requests.post(LINE_URL, headers=__headers, params=payload, files=__files)
        os.remove(self.img_file_path) # LINE後にjpgファイルは削除

    def close_camera(self): # カメラクローズ
        self.camera.close()

    def load_labels(self, path):
        # ラベルファイルをLoadして返す
        with open(path, 'r', encoding='utf-8') as f:
            __lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(__lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        return labels

    def set_input_tensor(self, interpreter, image):
        __tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(__tensor_index)()[0]
        input_tensor[:, :] = image
    
    def get_output_tensor(self, interpreter, index):
        __output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(__output_details['index']))
        return tensor

    def detect_objects(self, interpreter, labels, new_size, threshold, objs):
        # picameraの映像をキャプチャーしてTensorFlowでObjectDetection（物体検出）
        __stream = io.BytesIO()
        self.camera.capture(__stream, format='jpeg') # picameraの映像をjpegでキャプチャーする
        __stream.seek(0)
        __image = Image.open(__stream)
        __img_resize = __image.resize(new_size) # interpreterから読み込んだモデルのサイズにリサイズする
        self.set_input_tensor(interpreter, __img_resize)
        interpreter.invoke() # TensorFlowの呼び出し
        __classes = self.get_output_tensor(interpreter, 1) # クラス
        __scores = self.get_output_tensor(interpreter, 2) # 評価スコア
        __count = int(self.get_output_tensor(interpreter, 3)) # 評価数
        __results = []
        for i in range(__count): #scoreがthreshold（デフォルトで0.4）以上の物だけをフィルタリングしてresultsで返す
            if (__scores[i] >= threshold):
                __results.append(labels[__classes[i]])  # 検知ラベルを追加
                if len(objs) == 0: # 初回に認識したObjectをjpgで保存する（後でLINEで送付するため）
                    __time_stamp_file =  "{0:%Y-%m-%d-%H-%M-%S}".format(dt.datetime.now()) # 日付時刻をセット
                    self.img_file_path = SAVE_DIR + 'temp' + __time_stamp_file + '.jpg' # ディレクトリ、ファイル名をセット
                    __image.save(self.img_file_path, quality=25) # ファイルに保存しておく
        objs.extend(__results) # 検出されたオブジェクトを蓄える

#main
try:
    if __name__ == "__main__":
        os.chdir(os.path.dirname(os.path.abspath(__file__))) # カレントディレクトリをプログラムのあるディレクトリに移動する
        sec_camara = SecurityCameraClass()
        labels = sec_camara.load_labels(LABEL_FILE) # ラベル
        interpreter = Interpreter(MODEL_FILE) # TensorFlowモデル
        interpreter.allocate_tensors()
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
        new_size = (input_width, input_height) # モデルのサイズを読み込み
        __objs = [] # 映像から検出するオブジェクト
        rec = False # 録画中フラグ OFF
        start_detect = dt.datetime.now() # 検知開始時刻
        while True:
            rec_time = dt.timedelta(seconds=0)
            if GPIO.input(SC_CAMERA_PIN) == GPIO.HIGH: # 検知
                if rec == False: # 録画 OFFなら
                    __objs = []
                    start_detect = dt.datetime.now() # ビデオスタート時刻
                    rec = sec_camara.start_recordings() # 録画開始
                rec_time  = dt.datetime.now() - start_detect # 録画時間を計算
                if  rec_time.total_seconds() >= MAX_REC_TIME: # 録画最大時間を超えた時
                    rec = sec_camara.stop_recordings(list(set(__objs))) # 録画終了
                    start_detect = dt.datetime.now() # ビデオスタート時刻
                    rec = sec_camara.starat_recording() # 録画開始
            else: # 未検知
                if rec == True: # 録画 ON なら
                    rec = sec_camara.stop_recordings(list(set(__objs))) # 録画終了
                    start_detect = dt.datetime.now() # ビデオスタート時刻リセット
            if rec == True and len(__objs) == 0: # 録画中でObject未検出ならpicameの映像を元にObject Detection
                sec_camara.detect_objects(interpreter, labels, new_size, THRESHOLD, __objs)
            sleep(INTERVAL)
except KeyboardInterrupt:
    pass
GPIO.cleanup()
sec_camara.close_camera()