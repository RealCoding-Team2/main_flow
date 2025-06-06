
from flask import request, jsonify, current_app
from . import bp
import os
from app.llm_integration.llm import LLMRequester

@bp.route('/', methods=['POST']) # 엔드포인트 이름은 유지 (이미지/동영상 모두 처리)
def first_check():
    # 수신 받은 채팅내용
    recv = request.json.get('message')

    if recv is None:
        # 플라스크 문법에 따라 에러 return
        return jsonify({"error": "채팅내용이 없습니다!"}), 500
    
    llm_model_to_use = os.getenv("DEFAULT_MODEL")
    llm_requester = LLMRequester(model=llm_model_to_use)

    llm_text_response = llm_requester.send_message(
        message=recv,
        system_prompt="당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 정확한 답변을 제공해주세요."
    )

    return jsonify({"message": llm_text_response}), 200