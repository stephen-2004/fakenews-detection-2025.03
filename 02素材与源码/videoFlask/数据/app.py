import logging
import os

from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
app.secret_key = '3km8dz3U5Yst'
app.config['UPLOADED_VIDEOS_DEST'] = 'static/videos'
VIDEO_UPLOAD_FOLDER = 'static/videos'
TXT_SAVE_FOLDER = 'static/txt/video.txt'
RESULT_SAVE_FOLDER = 'static/result'

UPLOAD_FOLDERS = {
    'mp4': 'static/videos',
    'json': 'static/filejson',
    # 添加其他文件格式和对应的目录
}
from testmain import main

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/service')
def service():
    return render_template('service.html')


@app.route('/project')
def project():
    return render_template('project.html')


@app.route('/feature')
def feature():
    return render_template('feature.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/404')
def e404():
    return render_template('404.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    username = session.get('username')
    if username is None:
        return redirect('/')
    # from testmain import main
    files = request.files.getlist('files')
    if not files:
        return {'code': 1, 'message': '没有文件被上传'}, 400
    # 生成保存的文件名，防止同名覆盖
    filenames = []
    for file in files:
        filename = file.filename
        #print(filename)
        # video_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename)
        allowed_extensions = {'mp4'}
        if file.filename.lower().endswith(tuple(allowed_extensions)):
            filenames.append(filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        # 根据文件扩展名选择存储目录
        if file_ext in UPLOAD_FOLDERS:
            file.save(os.path.join(UPLOAD_FOLDERS[file_ext],  filename))

        # 保存文件
        # file.save(video_path)
    with open(TXT_SAVE_FOLDER, 'w') as file_list:
        for name in filenames:
            file_list.write(name.split('.')[0] + '\n')
        # 返回成功响应
    data=main.getpred("static/txt/video.txt")
    # print(data)
    #data = [1, 0, 1]
    # data_str = ','.join(str(x) for x in data)
    result_path = os.path.join(RESULT_SAVE_FOLDER, username, 'result.txt')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, 'a') as f:
        for name, result in zip(filenames, data):
            result_str = '真' if result == 0 else '假'
            f.write(f"{name.split('.')[0]} {result_str}\n")

    bar_data = len(filenames)
    pie_data_0 = data.count(0)
    pie_data_1 = data.count(1)

    combined_data = list(zip(filenames, data))


    return {'code': 0, 'message': '上传成功', 'data': data, 'bar_data': bar_data, 'pie_data_0': pie_data_0, 'pie_data_1': pie_data_1, 'combined_data': combined_data}, 200


# 用户信息文件
user_file = 'users.txt'


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirmPassword = request.form['confirmPassword']
        if password != confirmPassword:
            return render_template('register.html', error='两次密码不一致！',
                                   password=password, username=username,
                                   confirmPassword=confirmPassword)
        users = read_users()
        if username in users:
            return render_template('register.html', error='用户已存在',
                                   password=password, username=username,
                                   confirmPassword=confirmPassword)
        else:
            write_user(username, password)
            return redirect('/')
    return render_template('register.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = read_users()
        if username in users and users[username] == password:
            session['username'] = username
            return redirect('/index')
        else:
            return render_template('login.html', error="用户名或者密码不正确！")
    return render_template('login.html')


def read_users():
    users = {}
    try:
        with open(user_file, 'r') as f:
            for line in f:
                username, password = line.strip().split(',')
                users[username] = password
    except FileNotFoundError:
        pass
    return users


def write_user(username, password):
    with open(user_file, 'a') as f:
        f.write(f'{username},{password}\n')


if __name__ == '__main__':
    app.run(
        port=5001
    )
