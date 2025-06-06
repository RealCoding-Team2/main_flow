
from flask import request, jsonify, current_app
from . import bp
from app.llm_integration.llm import LLMRequester

@bp.route('/', methods=['POST']) # 엔드포인트 이름은 유지 (이미지/동영상 모두 처리)
def first_check():
    # 수신 받은 채팅내용
    recv = request.json.get('message')

    if recv is None:
        # 플라스크 문법에 따라 에러 return
        return jsonify({"error": "채팅내용이 없습니다!"}), 500
        

    return jsonify({"message": f'"{recv}"를 수신받았음~'}), 200