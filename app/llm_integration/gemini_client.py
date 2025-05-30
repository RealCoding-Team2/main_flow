
import requests
import json
from flask import current_app

# 딥페이크 분석을 위한 Gemini API 호출 함수
def call_gemini_api(prompt_text):
    api_url = current_app.config.get('GEMINI_API_URL')
    if not api_url:
        current_app.logger.error("Gemini API URL이 설정되지 않았습니다 (gemini_client - 딥페이크 분석용). LLM 연동이 불가능합니다.")
        return {"error": "LLM 서비스 설정 오류 (API URL 없음)"}, 500

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
    generation_config = {
        "temperature": 0.6,
        "topP": 0.9,
        "topK": 1,
        "maxOutputTokens": 2048,
        "responseMimeType": "application/json", # Gemini API가 직접 JSON 형식으로 응답하도록 요청
    }

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "safetySettings": safety_settings,
        "generationConfig": generation_config
    }
    headers = {"Content-Type": "application/json"}

    try:
        log_api_url = api_url.split('key=')[0] + 'key=*****' if 'key=' in api_url else api_url
        current_app.logger.debug(f"Gemini API 요청 시작 (딥페이크 분석용): {log_api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        llm_response_data = response.json()
        current_app.logger.debug("Gemini API로부터 성공적인 응답 수신 (딥페이크 분석용, JSON 파싱됨).")

        if 'candidates' in llm_response_data and llm_response_data['candidates']:
            first_candidate = llm_response_data['candidates'][0]
            if 'content' in first_candidate and 'parts' in first_candidate['content'] and first_candidate['content']['parts']:
                part_content = first_candidate['content']['parts'][0]
                if isinstance(part_content, dict):
                    current_app.logger.info("Gemini API가 직접 파싱된 JSON 객체를 반환했습니다 (딥페이크 분석용).")
                    return part_content, 200
                elif 'text' in part_content: # text 필드에 JSON "문자열"이 있는 경우
                    try:
                        parsed_json = json.loads(part_content['text'])
                        current_app.logger.info("Gemini API 응답의 text 필드에서 JSON을 성공적으로 파싱했습니다 (딥페이크 분석용).")
                        return parsed_json, 200
                    except json.JSONDecodeError as e:
                        current_app.logger.error(f"LLM 응답 내 JSON 문자열 파싱 중 오류 발생 (딥페이크 분석용): {e}")
                        return {"error": "LLM 응답 JSON 파싱 오류", "raw_text": part_content['text']}, 500
        
        if 'promptFeedback' in llm_response_data and 'blockReason' in llm_response_data['promptFeedback']:
            block_reason = llm_response_data['promptFeedback']['blockReason']
            current_app.logger.warning(f"LLM 요청이 안전상의 이유로 차단됨 (딥페이크 분석용): {block_reason}")
            return {"error": f"LLM 요청 차단됨: {block_reason}", "details": llm_response_data.get('promptFeedback')}, 400

        current_app.logger.error(f"LLM 응답에서 유효한 'candidates' 데이터를 찾을 수 없음 (딥페이크 분석용): {llm_response_data}")
        return {"error": "LLM으로부터 유효한 콘텐츠를 받지 못했습니다.", "raw_response": llm_response_data}, 500
    except requests.exceptions.Timeout:
        current_app.logger.error("Gemini API 호출 시간 초과 (딥페이크 분석용)")
        return {"error": "LLM API 호출 시간 초과"}, 504
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Gemini API 호출 중 네트워크 또는 HTTP 오류 발생 (딥페이크 분석용): {e}")
        if e.response is not None:
            error_response_text = e.response.text[:500] if hasattr(e.response, 'text') else "응답 내용 없음"
            current_app.logger.error(f"API 오류 응답 상태 코드: {e.response.status_code}, 내용 (일부): {error_response_text}...")
            try:
                error_details = e.response.json()
                return {"error": f"LLM API HTTP 오류: {str(e)}", "details": error_details}, e.response.status_code
            except json.JSONDecodeError:
                 return {"error": f"LLM API HTTP 오류: {str(e)}", "raw_error_response": error_response_text}, e.response.status_code if hasattr(e.response, 'status_code') else 502
        return {"error": f"LLM API 통신 오류 (응답 없음): {str(e)}"}, 502
    except Exception as e:
        current_app.logger.critical(f"LLM 호출 함수 실행 중 심각한 내부 오류 발생 (딥페이크 분석용): {e}", exc_info=True)
        return {"error": f"LLM 호출 중 예기치 않은 내부 오류 발생: {str(e)}"}, 500

# 일반 텍스트 채팅을 위한 Gemini API 호출 함수
def call_gemini_for_text_chat(user_message, conversation_history=None):
    api_url = current_app.config.get('GEMINI_API_URL')
    if not api_url:
        current_app.logger.error("Gemini API URL이 설정되지 않았습니다 (텍스트 채팅용). LLM 연동이 불가능합니다.")
        return {"error": "LLM 서비스 설정 오류 (API URL 없음)"}, 500

    prompt_for_chat = f"사용자: {user_message}\n챗봇:"

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
    generation_config = {
        "temperature": 0.7,
        "topP": 0.95,
        "topK": 40,
        "maxOutputTokens": 1024,
    }

    payload = {
        "contents": [{"parts": [{"text": prompt_for_chat}]}],
        "safetySettings": safety_settings,
        "generationConfig": generation_config
    }
    headers = {"Content-Type": "application/json"}

    try:
        log_api_url = api_url.split('key=')[0] + 'key=*****' if 'key=' in api_url else api_url
        current_app.logger.debug(f"Gemini API 요청 시작 (텍스트 채팅용): {log_api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        llm_response_data = response.json()
        current_app.logger.debug("Gemini API로부터 성공적인 응답 수신 (텍스트 채팅용).")

        if 'candidates' in llm_response_data and llm_response_data['candidates']:
            first_candidate = llm_response_data['candidates'][0]
            if 'content' in first_candidate and 'parts' in first_candidate['content'] and first_candidate['content']['parts']:
                if 'text' in first_candidate['content']['parts'][0]:
                    ai_text_response = first_candidate['content']['parts'][0]['text']
                    current_app.logger.info("Gemini API로부터 텍스트 응답을 성공적으로 받았습니다.")
                    return {"message": ai_text_response.strip()}, 200
        
        if 'promptFeedback' in llm_response_data and 'blockReason' in llm_response_data['promptFeedback']:
            block_reason = llm_response_data['promptFeedback']['blockReason']
            current_app.logger.warning(f"LLM 요청이 안전상의 이유로 차단됨 (텍스트 채팅용): {block_reason}")
            return {"error": f"LLM 요청 차단됨: {block_reason}", "details": llm_response_data.get('promptFeedback')}, 400

        current_app.logger.error(f"LLM 응답에서 유효한 텍스트 콘텐츠를 찾을 수 없음 (텍스트 채팅용): {llm_response_data}")
        return {"error": "LLM으로부터 유효한 텍스트 응답을 받지 못했습니다.", "raw_response": llm_response_data}, 500
    except requests.exceptions.Timeout:
        current_app.logger.error("Gemini API 호출 시간 초과 (텍스트 채팅용)")
        return {"error": "LLM API 호출 시간 초과"}, 504
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Gemini API 호출 중 네트워크 또는 HTTP 오류 발생 (텍스트 채팅용): {e}")
        if e.response is not None:
            error_response_text = e.response.text[:500] if hasattr(e.response, 'text') else "응답 내용 없음"
            current_app.logger.error(f"API 오류 응답 상태 코드: {e.response.status_code}, 내용 (일부): {error_response_text}...")
            try:
                error_details = e.response.json()
                return {"error": f"LLM API HTTP 오류: {str(e)}", "details": error_details}, e.response.status_code
            except json.JSONDecodeError:
                 return {"error": f"LLM API HTTP 오류: {str(e)}", "raw_error_response": error_response_text}, e.response.status_code if hasattr(e.response, 'status_code') else 502
        return {"error": f"LLM API 통신 오류 (응답 없음): {str(e)}"}, 502
    except Exception as e:
        current_app.logger.critical(f"LLM 텍스트 채팅 호출 함수 실행 중 심각한 내부 오류 발생: {e}", exc_info=True)
        return {"error": f"LLM 텍스트 채팅 호출 중 예기치 않은 내부 오류 발생: {str(e)}"}, 500