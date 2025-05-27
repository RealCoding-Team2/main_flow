from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# 클라이언트에서 이미지(jpg)를 업로드할 때 이 라우터가 동작합니다.
@app.route('/upload', methods=['POST'])
def upload_image_with_description():
    image = request.files.get('image')  # image라는 key값으로 파일데이터를 파싱

    # 없을 경우 오류코드 반환
    if not image:
        return jsonify({'error': '이미지 또는 설명이 없습니다.'}), 400

    #
    # 파일이 정상적으로 있을 경우 여기 부분부터 코드를 넣으면 됩니다.
    # 일단 예제로 받은 파일을 그대로 다시 저장하는 코드를 넣어봤습니다.
    #
    image.save(os.path.join(UPLOAD_FOLDER, image.filename))

    # 이미지의 상대 경로 URL 반환
    image_url = url_for('uploaded_file', filename=image.filename)

    # 처리 결과를 클라이언트에 반환합니다
    # url값은 서버에 저장된 이미지 파일 경로입니다.
    # 클라이언트에서는 해당 url값을 토대로 서버에 이미지 요청을 할 겁니다.
    return jsonify({
        'message': '업로드 성공!',
        'filename': image.filename,
        'url': image_url
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("./uploads", filename)

if __name__ == '__main__':
    app.run(debug=True)
