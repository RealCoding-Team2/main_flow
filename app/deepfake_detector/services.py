import cv2
import dlib
import numpy as np
from PIL import Image
import io
import json
import os
from flask import current_app
# from app.llm_integration.gemini_client import call_gemini_api
from app.llm_integration.llm import LLMRequester # <--- LLMRequester 클래스 임포트
import requests
from pathlib import Path

# Dlib 객체 로드 및 캐싱 
def get_dlib_objects_with_caching():
    if not hasattr(current_app, 'dlib_detector_instance') or \
       not hasattr(current_app, 'dlib_predictor_instance') or \
       current_app.dlib_detector_instance is None or \
       current_app.dlib_predictor_instance is None:
        
        model_path = current_app.config.get('DLIB_LANDMARK_MODEL_PATH')
        if not model_path or not os.path.exists(model_path):
            current_app.logger.error(f"Dlib 모델 파일을 찾을 수 없습니다. 설정된 경로: {model_path}")
            current_app.dlib_detector_instance = None
            current_app.dlib_predictor_instance = None
            return None, None
        try:
            current_app.logger.info(f"Dlib 모델 로드를 시작합니다. 경로: {model_path}")
            current_app.dlib_detector_instance = dlib.get_frontal_face_detector()
            current_app.dlib_predictor_instance = dlib.shape_predictor(model_path)
            current_app.logger.info("Dlib 모델 로드 및 앱 컨텍스트 캐싱 완료.")
        except Exception as e:
            current_app.logger.critical(f"Dlib 모델 로드 중 심각한 오류 발생: {e}", exc_info=True)
            current_app.dlib_detector_instance = None
            current_app.dlib_predictor_instance = None
            return None, None
    return current_app.dlib_detector_instance, current_app.dlib_predictor_instance

# 눈 깜빡임 분석 (이전과 동일)
def analyze_eye_blinking_from_landmarks(landmarks):
    if landmarks is None:
        return {"eye_blinking_status": "얼굴 랜드마크 데이터 없음"}
    try:
        def eye_aspect_ratio(eye_landmarks_pts):
            A = np.linalg.norm(eye_landmarks_pts[1] - eye_landmarks_pts[5])
            B = np.linalg.norm(eye_landmarks_pts[2] - eye_landmarks_pts[4])
            C = np.linalg.norm(eye_landmarks_pts[0] - eye_landmarks_pts[3])
            if C == 0: return 0.0
            ear = (A + B) / (2.0 * C)
            return ear

        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        status = "분석 필요 (단일 프레임 EAR)"
        if avg_ear < 0.18: status = "눈 감았을 가능성 매우 높음 (EAR 매우 낮음)"
        elif avg_ear < 0.22: status = "눈 약간 감았거나 깜빡이는 중일 가능성 (EAR 낮음)"
        elif avg_ear > 0.28: status = "눈 떴을 가능성 (EAR 보통)"
        else: status = "눈 깜빡임 보통 (EAR 일반 범위)"
        
        return {
            "left_eye_ear": round(float(left_ear), 4),
            "right_eye_ear": round(float(right_ear), 4),
            "average_ear": round(float(avg_ear), 4),
            "eye_blinking_status": status
        }
    except Exception as e:
        current_app.logger.error(f"눈 깜빡임 분석 중 오류 발생: {e}", exc_info=True)
        return {"eye_blinking_status": f"눈 깜빡임 분석 중 오류: {str(e)}"}

# 얼굴 일관성 분석 
def analyze_facial_consistency_from_landmarks(landmarks):
    if landmarks is None:
        return {"facial_consistency_status": "얼굴 랜드마크 데이터 없음"}
    try:
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        chin_tip = (landmarks.part(8).x, landmarks.part(8).y)
        left_eye_inner_corner = (landmarks.part(39).x, landmarks.part(39).y)
        right_eye_inner_corner = (landmarks.part(42).x, landmarks.part(42).y)
        
        nose_to_chin_distance = np.linalg.norm(np.array(nose_tip) - np.array(chin_tip))
        inter_pupillary_distance_approx = np.linalg.norm(np.array(left_eye_inner_corner) - np.array(right_eye_inner_corner))
        face_bounding_rect = landmarks.rect
        face_center_x = face_bounding_rect.left() + face_bounding_rect.width() / 2
        nose_symmetry_offset = abs(nose_tip[0] - face_center_x)

        return {
            "nose_to_chin_distance": round(float(nose_to_chin_distance), 2),
            "inter_pupillary_distance_approx": round(float(inter_pupillary_distance_approx), 2),
            "nose_symmetry_offset": round(float(nose_symmetry_offset), 2),
            "facial_consistency_status": "분석 완료"
        }
    except Exception as e:
        current_app.logger.error(f"얼굴 일관성 분석 중 오류 발생: {e}", exc_info=True)
        return {"facial_consistency_status": f"얼굴 일관성 분석 중 오류: {str(e)}"}

# 이미지 바이트(단일 프레임)로부터 딥페이크 관련 특징 추출 (이전과 동일)
def extract_single_frame_features(image_bytes):
    detector, predictor = get_dlib_objects_with_caching()
    if not detector or not predictor:
        current_app.logger.error("Dlib 객체를 사용할 수 없어 특징 추출을 진행할 수 없습니다.")
        return None, "Dlib 모델 초기화 실패 또는 사용 불가"
    
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image_pil)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        current_app.logger.error(f"이미지 파일 처리 중 오류 발생 (단일 프레임): {e}", exc_info=True)
        return None, f"이미지 파일 처리 오류: {str(e)}"

    detected_faces = detector(gray_image)
    if not detected_faces:
        current_app.logger.info("제공된 프레임에서 얼굴을 감지할 수 없습니다.")
        return {"face_detected": False, "error": "프레임에서 얼굴을 감지할 수 없습니다."}, "얼굴 미검출"

    main_face_landmarks = predictor(gray_image, detected_faces[0])
    eye_blinking_features = analyze_eye_blinking_from_landmarks(main_face_landmarks)
    facial_consistency_features = analyze_facial_consistency_from_landmarks(main_face_landmarks)

    face_rect = detected_faces[0]
    extracted_features_data = {
        "face_detected": True,
        "bounding_box": {
            "x": face_rect.left(), "y": face_rect.top(),
            "w": face_rect.width(), "h": face_rect.height()
        },
        "eye_blinking_analysis": eye_blinking_features,
        "facial_consistency_analysis": facial_consistency_features,
    }
    current_app.logger.info("단일 프레임으로부터 특징 추출 성공.")
    return extracted_features_data, "특징 추출 성공"

# 추출된 특징들(여러 프레임 또는 단일 이미지)과 사용자 상황 설명을 바탕으로 LLM에게 최종 판단 요청
def get_llm_deepfake_judgment(analysis_input_data, situation_description="", input_type="image"):
    try:
        llm_model_to_use = os.getenv("DEFAULT_MODEL") or "gpt-3.5-turbo" # .env 또는 기본값
        llm_requester = LLMRequester(model=llm_model_to_use) # 여기서 API 키 없으면 ValueError 발생
        current_app.logger.info(f"OpenAI LLM ({llm_model_to_use}) 객체 생성 완료.")
    except ValueError as ve: # LLMRequester가 API 키 없으면 ValueError 발생시킴
        current_app.logger.error(f"LLMRequester 초기화 오류 (아마도 API 키 부재): {ve}")
        return {
            "deepfake_probability": "판단 불가 (LLM API 키 설정 오류)",
            "reasoning": f"내부 LLM 서비스 초기화 중 오류가 발생했습니다: {str(ve)}",
            "recommendations_for_user": "관리자에게 문의하여 LLM 설정을 확인해주세요."
        }, 500 # 서버 설정 오류이므로 500 반환 고려
    except Exception as e_init: # 기타 초기화 오류
        current_app.logger.error(f"LLMRequester 초기화 중 예상치 못한 오류: {e_init}", exc_info=True)
        return {"error": f"LLM 서비스 초기화 중 심각한 오류: {str(e_init)}"}, 500
    

    system_prompt_for_openai = """당신은 매우 정확하고 객관적인 딥페이크 탐지 AI 전문가입니다.
제공된 시각적 특징 분석 데이터와 사용자 상황 설명을 바탕으로, 입력된 미디어가 딥페이크일 가능성을 평가해주세요.
응답은 반드시 다음 명시된 JSON 형식만을 사용해야 하며, 다른 어떤 텍스트도 포함해서는 안 됩니다.
JSON의 각 필드 값은 분석에 기반한 구체적인 내용이어야 합니다.

[필수 JSON 응답 형식]
{
  "deepfake_probability": "string (다음 중 하나: 매우 높음, 높음, 중간, 낮음, 매우 낮음, 판단 어려움)",
  "confidence_score": "float (0.0에서 1.0 사이의 판단 신뢰도 점수)",
  "reasoning": "string (판단 근거를 상세히 설명. 어떤 특징이 의심을 증폭시키거나 감소시키는지 명시. 정지 이미지와 동영상의 특징을 구분하여 설명.)",
  "key_indicators": ["string (딥페이크 판단에 결정적이었던 주요 지표들 목록)"],
  "recommendations_for_user": "string (사용자를 위한 구체적인 다음 단계 조언이나 주의사항)"
}
"""
    # 사용자 메시지 구성 
    input_type_description = "정지된 이미지" if input_type == "image" else "동영상"
    user_message_content = f"""
[분석 대상 정보]
- 입력 타입: {input_type_description}

[분석 데이터 요약]
{json.dumps(analysis_input_data, indent=2, ensure_ascii=False)}

[사용자 상황 설명]
{situation_description if situation_description else "제공되지 않음"}

위 정보를 바탕으로 JSON 형식에 맞춰 딥페이크 분석 결과를 제공해주세요.
"""

    try:
        # yongs3님의 LLMRequester 클래스 인스턴스 생성
        # .env 파일의 DEFAULT_MODEL 또는 기본 모델명(예: "gpt-3.5-turbo") 사용 가능
        # API 키는 LLMRequester 내부에서 .env 통해 로드됨
        llm_model_to_use = os.getenv("DEFAULT_MODEL")
        llm_requester = LLMRequester(model=llm_model_to_use)

        # LLMRequester의 send_message 메서드 호출
        # 추가 파라미터 (temperature, max_tokens 등)는 LLMRequester의 기본값 또는 환경 변수 설정을 따르거나,
        # 필요시 여기서 직접 **kwargs로 전달할 수 있습니다.
        # 예: llm_requester.send_message(user_message_content, system_prompt=system_prompt_for_openai, temperature=0.5)
        llm_text_response = llm_requester.send_message(
            message=user_message_content,
            system_prompt=system_prompt_for_openai
        )
        current_app.logger.debug(f"OpenAI LLM 원시 응답: {llm_text_response}")

        # LLM 응답이 JSON 문자열이라고 가정하고 파싱
        # (만약 LLMRequester가 이미 딕셔너리를 반환한다면 json.loads는 필요 없음)
        try:
            parsed_llm_response = json.loads(llm_text_response)
            # OpenAI 응답이 위에서 정의한 JSON 형식을 잘 따랐는지 간단히 확인 가능
            if not all(key in parsed_llm_response for key in ["deepfake_probability", "reasoning", "recommendations_for_user"]):
                current_app.logger.warning(f"LLM 응답이 예상된 JSON 형식을 완전히 따르지 않음: {parsed_llm_response}")
                # 필수 키가 없는 경우, 일부 기본값을 채워넣거나 오류로 처리할 수 있음
            return parsed_llm_response, 200
        except json.JSONDecodeError as e:
            current_app.logger.error(f"OpenAI LLM 응답 JSON 파싱 오류: {e} - 응답 내용: {llm_text_response}")
            # LLM이 JSON 형식이 아닌 일반 텍스트로 답변했을 가능성
            return {"error": "LLM 응답 형식 오류 (JSON 파싱 실패)", "raw_response": llm_text_response}, 500
        except TypeError as te: # 만약 llm_text_response가 이미 객체인데 json.loads 하려 할 경우
             current_app.logger.error(f"OpenAI LLM 응답 타입 오류 (이미 객체일 수 있음): {te} - 응답: {llm_text_response}")
             if isinstance(llm_text_response, dict): # 이미 딕셔너리라면 그대로 반환
                 return llm_text_response, 200
             return {"error": "LLM 응답 타입 오류", "raw_response": str(llm_text_response)}, 500


    except ValueError as ve: # LLMRequester 초기화 시 API 키 없음 등의 오류
        current_app.logger.error(f"LLMRequester 초기화 오류: {ve}")
        return {"error": f"LLM 서비스 설정 오류: {ve}"}, 500
    except Exception as e: # 그 외 API 호출 중 발생할 수 있는 예외
        current_app.logger.critical(f"LLM 판단 중 예외 발생: {e}", exc_info=True)
        return {"error": f"LLM 판단 중 오류 발생: {str(e)}"}, 500

# fast-stt API 호출을 위한 함수
def call_fast_stt_api(audio_file_path):
    """
    fast-stt 서비스에 음성 파일을 전송하여 STT 변환을 수행합니다.
    
    Args:
        audio_file_path (str): 음성 파일의 경로
        
    Returns:
        dict: STT 결과 또는 에러 정보
    """
    try:
        # fast-stt 서비스 URL (환경변수에서 가져오거나 기본값 사용)
        stt_service_url = os.getenv('FAST_STT_SERVICE_URL', 'http://localhost:8001')
        stt_endpoint = f"{stt_service_url}/api/transcribe"
        
        current_app.logger.info(f"fast-stt 서비스 호출 시작: {stt_endpoint}")
        
        # 음성 파일을 multipart/form-data로 전송
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': (Path(audio_file_path).name, audio_file, 'audio/mpeg')}
            
            # POST 요청으로 STT 서비스 호출
            response = requests.post(
                stt_endpoint,
                files=files,
                timeout=60  # 60초 타임아웃
            )
        
        if response.status_code == 200:
            stt_result = response.json()
            current_app.logger.info("fast-stt 서비스 호출 성공")
            return {
                "success": True,
                "transcription": stt_result.get("transcription", ""),
                "language": stt_result.get("language", "ko"),
                "confidence": stt_result.get("confidence", 0.0)
            }
        else:
            current_app.logger.error(f"fast-stt 서비스 오류: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"STT 서비스 오류 (HTTP {response.status_code}): {response.text}"
            }
            
    except requests.exceptions.Timeout:
        current_app.logger.error("fast-stt 서비스 타임아웃")
        return {
            "success": False,
            "error": "STT 서비스 응답 시간 초과 (60초)"
        }
    except requests.exceptions.ConnectionError:
        current_app.logger.error("fast-stt 서비스 연결 실패")
        return {
            "success": False,
            "error": "STT 서비스에 연결할 수 없습니다. 서비스가 실행 중인지 확인하세요."
        }
    except Exception as e:
        current_app.logger.error(f"fast-stt 서비스 호출 중 예외 발생: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"STT 서비스 호출 중 오류: {str(e)}"
        }

def analyze_text_for_voicephishing(text, situation=""):
    """
    텍스트 내용을 분석하여 보이스피싱 여부를 판단합니다.
    
    Args:
        text (str): 분석할 텍스트
        situation (str): 사용자가 제공한 상황 설명
        
    Returns:
        dict: 보이스피싱 분석 결과
    """
    try:
        llm_model_to_use = os.getenv("DEFAULT_MODEL") or "gpt-3.5-turbo"
        llm_requester = LLMRequester(model=llm_model_to_use)
        current_app.logger.info(f"보이스피싱 분석을 위한 LLM ({llm_model_to_use}) 객체 생성 완료.")
    except ValueError as ve:
        current_app.logger.error(f"LLMRequester 초기화 오류: {ve}")
        return {
            "voicephishing_probability": "판단 불가 (LLM API 키 설정 오류)",
            "reasoning": f"내부 LLM 서비스 초기화 중 오류가 발생했습니다: {str(ve)}",
            "recommendations_for_user": "관리자에게 문의하여 LLM 설정을 확인해주세요."
        }
    except Exception as e:
        current_app.logger.error(f"LLM 초기화 중 예상치 못한 오류: {e}", exc_info=True)
        return {
            "voicephishing_probability": "판단 불가 (서버 오류)",
            "reasoning": f"LLM 서비스 초기화 중 심각한 오류: {str(e)}",
            "recommendations_for_user": "잠시 후 다시 시도해주세요."
        }

    system_prompt = """당신은 보이스피싱 탐지 전문가입니다.
제공된 텍스트 내용을 분석하여 보이스피싱(전화금융사기)일 가능성을 평가해주세요.

다음과 같은 보이스피싱 특징들을 고려하세요:
1. 긴급성 강조 ("즉시", "지금 당장", "빨리")
2. 금융 관련 용어 (계좌, 송금, 대출, 카드, 보험금)
3. 신분 사칭 (경찰, 검찰, 은행원, 금융감독원)
4. 개인정보 요구 (주민번호, 계좌번호, 비밀번호)
5. 위협적 언어 ("체포", "압류", "수사", "법적 조치")
6. 의심스러운 요구 (ATM 조작, 앱 설치, 화면 공유)

응답은 반드시 다음 JSON 형식을 따라주세요:
{
  "voicephishing_probability": "string (매우 높음/높음/중간/낮음/매우 낮음 중 하나)",
  "confidence_score": "float (0.0-1.0 사이의 신뢰도)",
  "reasoning": "string (구체적인 판단 근거)",
  "detected_patterns": ["보이스피싱으로 의심되는 패턴들"],
  "risk_level": "string (고위험/중위험/저위험/안전 중 하나)",
  "recommendations_for_user": "string (사용자를 위한 구체적인 조치 방안)"
}"""

    user_message = f"""
[분석할 텍스트]
{text}

[상황 설명]
{situation if situation else "제공되지 않음"}

위 텍스트를 분석하여 보이스피싱 여부를 판단해주세요.
"""

    try:
        llm_response = llm_requester.send_message(
            message=user_message,
            system_prompt=system_prompt
        )
        current_app.logger.debug(f"보이스피싱 분석 LLM 응답: {llm_response}")

        # JSON 응답 파싱
        if isinstance(llm_response, str):
            analysis_result = json.loads(llm_response)
        else:
            analysis_result = llm_response
            
        current_app.logger.info("보이스피싱 분석 완료")
        return analysis_result

    except json.JSONDecodeError as je:
        current_app.logger.error(f"LLM 응답 JSON 파싱 오류: {je}")
        return {
            "voicephishing_probability": "판단 불가 (응답 파싱 오류)",
            "reasoning": "AI 응답을 처리하는 중 오류가 발생했습니다.",
            "recommendations_for_user": "다시 시도해주시거나 관리자에게 문의해주세요."
        }
    except Exception as e:
        current_app.logger.error(f"보이스피싱 분석 중 오류: {e}", exc_info=True)
        return {
            "voicephishing_probability": "판단 불가 (서버 오류)",
            "reasoning": f"분석 중 오류가 발생했습니다: {str(e)}",
            "recommendations_for_user": "잠시 후 다시 시도해주세요."
        }

def analyze_audio_for_voicephishing(audio_file_path, situation=""):
    """
    음성 파일을 STT로 변환하고 보이스피싱 여부를 분석합니다.
    
    Args:
        audio_file_path (str): 분석할 음성 파일 경로
        situation (str): 사용자가 제공한 상황 설명
        
    Returns:
        tuple: (stt_result, voicephishing_analysis)
    """
    current_app.logger.info(f"음성 파일 보이스피싱 분석 시작: {audio_file_path}")
    
    # 1. STT 변환
    stt_result = call_fast_stt_api(audio_file_path)
    
    if not stt_result.get("success", False):
        current_app.logger.error("STT 변환 실패")
        return stt_result, {
            "voicephishing_probability": "판단 불가 (음성 변환 실패)",
            "reasoning": stt_result.get("error", "음성을 텍스트로 변환할 수 없습니다."),
            "recommendations_for_user": "음성 파일이 명확한지 확인하고 다시 시도해주세요."
        }
    
    transcription = stt_result.get("transcription", "")
    if not transcription.strip():
        current_app.logger.warning("STT 결과가 비어있음")
        return stt_result, {
            "voicephishing_probability": "판단 불가 (음성 내용 없음)",
            "reasoning": "음성에서 인식된 텍스트가 없습니다.",
            "recommendations_for_user": "더 명확한 음성이 포함된 파일을 업로드해주세요."
        }
    
    # 2. 텍스트 내용 보이스피싱 분석
    voicephishing_analysis = analyze_text_for_voicephishing(transcription, situation)
    
    current_app.logger.info("음성 파일 보이스피싱 분석 완료")
    return stt_result, voicephishing_analysis