
from flask import request, jsonify, current_app
from . import bp
from app.llm_integration.llm import LLMRequester
import requests # requests 라이브러리가 필요하다면 임포트
import os

llm_model_to_use = os.getenv("DEFAULT_MODEL")
llm_requester = LLMRequester(model=llm_model_to_use)

USE_RAG = True  # 기본값은 RAG 사용

@bp.route('/', methods=['POST']) # 엔드포인트 이름은 유지 (이미지/동영상 모두 처리)
def first_check():
    global USE_RAG

    # 수신 받은 채팅내용
    recv = request.json.get('message')

    if recv is None:
        return jsonify({"error": "채팅내용이 없습니다!"}), 500
    
    if recv.strip().lower() == "//":
        recv = recv.replace("//", "")

        if USE_RAG: print("RAG 모드가 꺼졌습니다. 이후 질문은 LLM 단독 응답으로 처리됩니다.")
        else:       print("RAG 모드가 켜졌습니다. 이후 질문은 LLM 단독 응답으로 처리됩니다.")

        # USE_RAG 상태값 변경
        USE_RAG = not USE_RAG

        message_status = "켜져있습니다." if USE_RAG else "꺼져있습니다."
        
        return jsonify({"message": f"RAG 모드가 {message_status}"}), 200

    rag_response = None
    if USE_RAG:
        # 여기에서 RAG서버와 통신
        rag_server_url = "http://localhost:8000/search"
        query = {
            "query": recv
        }

        rag_response = get_rag_data(query, rag_server_url)


    final_question = recv
        
    if rag_response:
        final_question = build_summarized_prompt(recv, rag_response)

    llm_text_response = llm_requester.send_message(
        message=final_question,
        system_prompt="당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 정확한 답변을 제공해주세요."
    )

    return jsonify({"message": llm_text_response}), 200

def build_summarized_prompt(recv, rag_data):
    prompt = f"""
다음은 한 사용자가 보낸 질문입니다:
[사용자 질의]
{recv}

그리고 다음은 이 질의와 관련성이 높은 실제 뉴스 기사나 사례 정보를 요약한 내용입니다:
"""

    for idx, item in enumerate(rag_data["results"][:5]):  # 상위 5개 문서 사용
        title = item["metadata"].get("title", "제목 없음")
        date  = item["metadata"].get("date", "날짜 없음")
        url   = item["metadata"].get("url", "출처 없음")
        text  = item["text"]

        summary = summarize_text_with_llm(text)

        prompt += f"""
                    [문서 {idx+1}]
                    제목: {title}
                    날짜: {date}
                    출처: {url}
                    요약: {summary}
                    """
        print(prompt)

    prompt += """
위의 요약된 문서 정보를 참고하여, 사용자에게 친절하고 구체적으로 어떻게 대응하면 좋을지 조언해 주세요.

특히 유사 사례나 사기 수법이 있다면 구체적으로 경고해 주세요. 관련 없는 문서는 무시해도 됩니다.

또한 첨부한 관련 문서들 중 관련있는 문서들을 짧게(50자) 설명하여 이런 예시사례가 있다는 것을 강조하는데, 그 문서의 날짜나 출처도 같이 강조해서 설명해주세요.
"""

    return prompt

def summarize_text_with_llm(text):
    response = llm_requester.send_message(
        message=text,
        system_prompt="당신은 뉴스 기사나 보이스피싱 정보 문서를 250~300자 내외로 간결하고 핵심적인 요약을 하는 요약 전문가입니다."
    )

    return response

def get_rag_data(query, rag_server_url="http://localhost:8000/search"):
    rag_data = None

    try:
        rag_response = requests.post(rag_server_url, json=query)
        rag_response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        rag_data = rag_response.json()

    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"RAG 서버 통신 오류: {e}")

    except Exception as e:
        current_app.logger.error(f"RAG 응답 처리 중 오류: {e}")

    return rag_data