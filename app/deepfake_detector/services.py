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
from app.llm_integration.llm import LLMRequester
import pandas as pd

def get_dlib_objects_with_caching():
    if not hasattr(current_app, 'dlib_detector_instance') or not hasattr(current_app, 'dlib_predictor_instance') or current_app.dlib_detector_instance is None or current_app.dlib_predictor_instance is None:
        model_path = current_app.config.get('DLIB_LANDMARK_MODEL_PATH')
        if not model_path or not os.path.exists(model_path): return None, None
        try:
            current_app.dlib_detector_instance = dlib.get_frontal_face_detector(); current_app.dlib_predictor_instance = dlib.shape_predictor(model_path)
            return current_app.dlib_detector_instance, current_app.dlib_predictor_instance
        except Exception as e: return None, None
    return current_app.dlib_detector_instance, current_app.dlib_predictor_instance

def analyze_eye_blinking_from_landmarks(landmarks):
    if landmarks is None: return {"eye_blinking_status": "얼굴 랜드마크 데이터 없음"}
    try:
        def eye_aspect_ratio(eye_landmarks_pts):
            A=np.linalg.norm(eye_landmarks_pts[1]-eye_landmarks_pts[5]); B=np.linalg.norm(eye_landmarks_pts[2]-eye_landmarks_pts[4]); C=np.linalg.norm(eye_landmarks_pts[0]-eye_landmarks_pts[3])
            if C == 0: return 0.0
            return (A + B) / (2.0 * C)
        left_eye_pts=np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]); right_eye_pts=np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_ear, right_ear = eye_aspect_ratio(left_eye_pts), eye_aspect_ratio(right_eye_pts); avg_ear = (left_ear + right_ear) / 2.0; status = "눈 깜빡임 보통"
        if avg_ear < 0.2: status = "눈 감았을 가능성 높음"
        return {"average_ear": round(float(avg_ear), 4), "status": status}
    except Exception as e: return {"status": f"눈 깜빡임 분석 오류: {e}"}

def _analyze_head_pose_from_landmarks(landmarks, image_shape):
    if landmarks is None: return {"head_pose_status": "얼굴 랜드마크 데이터 없음"}
    try:
        model_points = np.array([(0.0,0.0,0.0), (0.0,-330.0,-65.0), (-225.0,170.0,-135.0), (225.0,170.0,-135.0), (-150.0,-150.0,-125.0), (150.0,-150.0,-125.0)])
        image_points = np.array([(landmarks.part(30).x,landmarks.part(30).y), (landmarks.part(8).x,landmarks.part(8).y), (landmarks.part(36).x,landmarks.part(36).y), (landmarks.part(45).x,landmarks.part(45).y), (landmarks.part(48).x,landmarks.part(48).y), (landmarks.part(54).x,landmarks.part(54).y)], dtype="double")
        focal_length = image_shape[1]; center = (image_shape[1]/2, image_shape[0]/2)
        camera_matrix = np.array([[focal_length,0,center[0]], [0,focal_length,center[1]], [0,0,1]], dtype="double"); dist_coeffs = np.zeros((4,1))
        (_, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector); projection_matrix = np.hstack((rotation_matrix, np.zeros((3,1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()[:3]
        return {"pitch": round(pitch,2), "yaw": round(yaw,2), "roll": round(roll,2), "status": "분석 완료"}
    except Exception as e: return {"status": f"머리 자세 추정 오류: {e}"}

def _stabilize_angle_series(angle_series, smoothing_window=3):
    if len(angle_series) < smoothing_window: return np.array(angle_series)
    unwrapped_angles = np.unwrap(np.deg2rad(angle_series)); unwrapped_degrees = np.rad2deg(unwrapped_angles)
    return pd.Series(unwrapped_degrees).rolling(window=smoothing_window, min_periods=1, center=True).mean().to_numpy()

def analyze_temporal_features(frame_features_list):
    if not frame_features_list or len(frame_features_list) < 5: return {"status": "분석할 프레임 부족"}
    yaws = [f.get('head_pose_analysis', {}).get('yaw', 0) for f in frame_features_list]; pitches = [f.get('head_pose_analysis', {}).get('pitch', 0) for f in frame_features_list]; rolls = [f.get('head_pose_analysis', {}).get('roll', 0) for f in frame_features_list]
    yaws_stable = _stabilize_angle_series(yaws); pitches_stable = _stabilize_angle_series(pitches); rolls_stable = _stabilize_angle_series(rolls)
    yaw_velocities = np.diff(yaws_stable); pitch_velocities = np.diff(pitches_stable); roll_velocities = np.diff(rolls_stable)
    return {
        "velocity_jitter_stable": {"yaw_velocity_std": round(np.std(yaw_velocities),2), "pitch_velocity_std": round(np.std(pitch_velocities),2), "roll_velocity_std": round(np.std(roll_velocities),2)},
        "range_of_motion_stable": {"yaw_range": round(np.max(yaws_stable)-np.min(yaws_stable),2), "pitch_range": round(np.max(pitches_stable)-np.min(pitches_stable),2), "roll_range": round(np.max(rolls_stable)-np.min(rolls_stable),2)},
    }

def extract_single_frame_features(image_bytes):
    detector, predictor = get_dlib_objects_with_caching()
    if not detector or not predictor: return None, "Dlib 모델 초기화 실패"
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB"); image_np = np.array(image_pil)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY); image_shape = image_np.shape
    except Exception as e: return None, f"이미지 처리 오류: {e}"
    detected_faces = detector(gray_image)
    if not detected_faces: return {"face_detected": False, "error": "프레임에서 얼굴 미검출"}, "얼굴 미검출"
    main_face_landmarks = predictor(gray_image, detected_faces[0]); face_rect = detected_faces[0]
    return {
        "face_detected": True, "bounding_box": {"x": face_rect.left(), "y": face_rect.top(), "w": face_rect.width(), "h": face_rect.height()},
        "eye_blinking_analysis": analyze_eye_blinking_from_landmarks(main_face_landmarks),
        "head_pose_analysis": _analyze_head_pose_from_landmarks(main_face_landmarks, image_shape),
    }, "특징 추출 성공"

# --- ✨새로운 최종 통합 함수: 마스터 AI✨ ---
def get_final_synthesized_response(deepfake_analysis_data, rag_search_results, user_situation):
    try:
        llm_model_to_use = os.getenv("DEFAULT_MODEL") # .env 또는 기본값
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
    

    system_prompt_for_master_ai = """당신은 딥페이크 기술 분석과 실제 피싱 사례 데이터베이스를 모두 활용하여 사용자에게 최종적인 조언을 제공하는 '피싱 대응 총괄 AI'입니다.
두 가지 보고서를 종합하여, 사용자에게 가장 도움이 되는 최종 답변을 하나의 JSON으로 생성해야 합니다.

[보고서 1: 딥페이크 기술 분석 결과]
영상 속 인물의 움직임을 분석하여 딥페이크일 확률을 기술적으로 평가한 내용입니다.

[보고서 2: RAG 시스템의 유사 사례 검색 결과]
사용자의 상황 설명과 관련된 실제 피싱 범죄 사례나 뉴스를 데이터베이스에서 검색한 결과입니다.

[최종 답변 생성 규칙]
1.  두 보고서의 내용을 모두 조합하여 `reasoning`을 작성하세요.
    - 예: "영상 분석 결과, 머리 움직임 속도가 부자연스러워 딥페이크일 확률이 '높음'으로 나타났습니다. 또한, 사용자의 상황 설명을 바탕으로 데이터베이스를 검색한 결과, 최근 '자녀 사칭' 수법이 유행하고 있어 각별한 주의가 필요합니다."
2.  `deepfake_probability`와 `confidence_score`는 [보고서 1]의 값을 기반으로 하되, [보고서 2]의 내용이 판단을 강화하면 값을 소폭 상향 조정할 수 있습니다.
3.  `recommendations_for_user`는 두 보고서의 내용을 바탕으로 가장 안전하고 구체적인 행동 요령을 제시하세요.

[필수 JSON 응답 형식]
{
  "deepfake_probability": "string",
  "confidence_score": "float",
  "reasoning": "string (두 보고서를 종합한 최종 판단 근거)",
  "recommendations_for_user": "string (가장 구체적이고 안전한 최종 권장 사항)"
}
"""
    user_message_for_master_ai = f"""[보고서 1: 딥페이크 기술 분석 결과]
{json.dumps(deepfake_analysis_data, indent=2, ensure_ascii=False)}

[보고서 2: RAG 시스템의 유사 사례 검색 결과]
{json.dumps(rag_search_results, indent=2, ensure_ascii=False)}

[사용자 상황 원본]
{user_situation}

위 보고서들을 바탕으로, 규칙에 맞게 최종 답변을 JSON 형식으로만 생성해주세요.
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
        stt_service_url = os.getenv('FAST_STT_SERVICE_URL', 'http://localhost:8000')
        stt_endpoint = f"{stt_service_url}/transcribe"
        
        current_app.logger.info(f"fast-stt 서비스 호출 시작: {stt_endpoint}")
        
        # 음성 파일을 multipart/form-data로 전송 (fast-stt API 규격에 맞게 'file' 키 사용)
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (Path(audio_file_path).name, audio_file, 'audio/mpeg')}
            
            # POST 요청으로 STT 서비스 호출
            response = requests.post(
                stt_endpoint,
                files=files,
                timeout=600  # 10분 타임아웃 (긴 음성 파일 처리 고려)
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

def summarize_text_for_analysis(text, max_length=1000):
    """
    긴 텍스트를 LLM을 통해 요약하여 핵심 내용만 추출합니다.
    
    Args:
        text (str): 요약할 원본 텍스트
        max_length (int): 원본 텍스트 길이 기준 (이보다 짧으면 요약하지 않음)
        
    Returns:
        dict: 요약 결과 또는 원본 텍스트
    """
    if len(text.strip()) <= max_length:
        current_app.logger.info(f"텍스트 길이가 {len(text)}자로 짧아 요약을 생략합니다.")
        return {
            "original_text": text,
            "summarized_text": text,
            "is_summarized": False,
            "summary_reason": "텍스트가 충분히 짧음"
        }
    
    try:
        llm_model_to_use = os.getenv("DEFAULT_MODEL")
        llm_requester = LLMRequester(model=llm_model_to_use)
        current_app.logger.info(f"텍스트 요약을 위한 LLM ({llm_model_to_use}) 객체 생성 완료.")
    except ValueError as ve:
        current_app.logger.error(f"텍스트 요약 LLM 초기화 오류: {ve}")

        return {
            "original_text": text,
            "summarized_text": text,
            "is_summarized": False,
            "summary_reason": f"LLM 초기화 실패: {str(ve)}"
        }
    except Exception as e:
        current_app.logger.error(f"텍스트 요약 LLM 초기화 중 예상치 못한 오류: {e}", exc_info=True)
        return {
            "original_text": text,
            "summarized_text": text,
            "is_summarized": False,
            "summary_reason": f"서버 오류: {str(e)}"
        }

    system_prompt = """당신은 통화 내용 요약 전문가입니다.
제공된 통화 텍스트를 분석하여 보이스피싱 탐지에 필요한 핵심 정보만을 추출해 요약해주세요.

다음 요소들을 중점적으로 포함해서 요약하세요:
1. 상대방이 자신을 누구라고 소개했는지 (신분, 소속)
2. 금융 관련 언급 사항 (계좌, 카드, 대출, 송금, 보험금 등)
3. 긴급성을 강조하는 표현들
4. 개인정보 요구 사항
5. 위협적이거나 강압적인 언어
6. 특정 행동을 요구하는 내용 (ATM 조작, 앱 설치, 계좌 이체 등)
7. 통화자의 목적과 의도

요약은 간결하면서도 보이스피싱 판단에 필요한 모든 정보를 포함해야 합니다.
원본 텍스트의 중요한 맥락과 뉘앙스를 유지해주세요."""

    user_message = f"""
다음 통화 내용을 보이스피싱 탐지 분석에 적합하도록 요약해주세요:

[원본 통화 내용]
{text}

위 내용을 핵심 정보만 추출하여 요약해주세요.
"""

    try:
        summary_response = llm_requester.send_message(
            message=user_message,
            system_prompt=system_prompt
        )
        current_app.logger.info("텍스트 요약 완료")
        
        return {
            "original_text": text,
            "summarized_text": summary_response.strip(),
            "is_summarized": True,
            "summary_reason": f"원본 길이 {len(text)}자에서 {len(summary_response)}자로 요약"
        }

    except Exception as e:
        current_app.logger.error(f"텍스트 요약 중 오류: {e}", exc_info=True)

        return {
            "original_text": text,
            "summarized_text": text,
            "is_summarized": False,
            "summary_reason": f"요약 실패: {str(e)}"
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
        llm_model_to_use = os.getenv("DEFAULT_MODEL")
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

**중요: 반드시 아래의 정확한 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.**

{
  "voicephishing_probability": "매우 높음",
  "confidence_score": 0.85,
  "reasoning": "구체적인 판단 근거",
  "detected_patterns": ["패턴1", "패턴2"],
  "risk_level": "고위험",
  "recommendations_for_user": "구체적인 조치 방안"
}

가능한 값들:
- voicephishing_probability: "매우 높음", "높음", "중간", "낮음", "매우 낮음"
- risk_level: "고위험", "중위험", "저위험", "안전"
- confidence_score: 0.0~1.0 사이의 숫자"""

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
        
        if isinstance(llm_response, str) and llm_response.startswith("오류가 발생했습니다:"):
            current_app.logger.error(f"LLM API 호출 실패: {llm_response}")
            return {
                "voicephishing_probability": "판단 불가 (LLM API 오류)",
                "confidence_score": 0.0,
                "reasoning": f"LLM 서비스와 통신 중 오류가 발생했습니다: {llm_response}",
                "detected_patterns": [],
                "risk_level": "분석 실패",
                "recommendations_for_user": "인터넷 연결을 확인하고 잠시 후 다시 시도해주세요."
            }

        try:
            if isinstance(llm_response, dict):
                analysis_result = llm_response
            elif isinstance(llm_response, str):
                llm_response = llm_response.strip()
                
                if llm_response.startswith('```json') and llm_response.endswith('```'):
                    llm_response = llm_response[7:-3].strip()
                elif llm_response.startswith('```') and llm_response.endswith('```'):
                    llm_response = llm_response[3:-3].strip()
                
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = llm_response[start_idx:end_idx+1]
                    analysis_result = json.loads(json_str)
                else:
                    analysis_result = {
                        "voicephishing_probability": "판단 불가 (응답 형식 오류)",
                        "confidence_score": 0.0,
                        "reasoning": f"LLM이 JSON 형식이 아닌 응답을 보냈습니다: {llm_response[:500]}",
                        "detected_patterns": [],
                        "risk_level": "분석 실패",
                        "recommendations_for_user": "시스템 관리자에게 문의해주세요."
                    }
            else:
                analysis_result = {
                    "voicephishing_probability": "판단 불가 (응답 타입 오류)",
                    "confidence_score": 0.0,
                    "reasoning": f"LLM 응답 타입이 예상과 다릅니다: {type(llm_response)}",
                    "detected_patterns": [],
                    "risk_level": "분석 실패",
                    "recommendations_for_user": "시스템 관리자에게 문의해주세요."
                }
                
        except json.JSONDecodeError as je:
            analysis_result = {
                "voicephishing_probability": "판단 불가 (JSON 파싱 실패)",
                "confidence_score": 0.0,
                "reasoning": f"AI 응답을 JSON으로 파싱할 수 없습니다. 오류: {str(je)}",
                "detected_patterns": [],
                "risk_level": "분석 실패",
                "recommendations_for_user": "다시 시도해주시거나 관리자에게 문의해주세요."
            }
            
        return analysis_result

    except Exception as e:
        current_app.logger.error(f"보이스피싱 분석 중 오류: {e}", exc_info=True)
        return {
            "voicephishing_probability": "판단 불가 (서버 오류)",
            "confidence_score": 0.0,
            "reasoning": f"분석 중 오류가 발생했습니다: {str(e)}",
            "detected_patterns": [],
            "risk_level": "분석 실패",
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
    
    # 2. 텍스트 요약 (새로 추가된 단계)
    current_app.logger.info("텍스트 요약 단계 시작")
    summary_result = summarize_text_for_analysis(transcription)
    
    # STT 결과에 요약 정보 추가
    stt_result["summary_info"] = {
        "is_summarized": summary_result["is_summarized"],
        "summary_reason": summary_result["summary_reason"],
        "original_length": len(summary_result["original_text"]),
        "summarized_length": len(summary_result["summarized_text"]),
        "summarized_text": summary_result["summarized_text"]
    }
    
    # 3. 요약된 텍스트로 보이스피싱 분석
    text_to_analyze = summary_result["summarized_text"]
    current_app.logger.info(f"보이스피싱 분석할 텍스트 길이: {len(text_to_analyze)}자")
    
    voicephishing_analysis = analyze_text_for_voicephishing(text_to_analyze, situation)
    
    # 분석 결과에 요약 정보 추가
    if isinstance(voicephishing_analysis, dict):
        voicephishing_analysis["text_processing"] = {
            "was_summarized": summary_result["is_summarized"],
            "original_text_length": len(transcription),
            "analyzed_text_length": len(text_to_analyze)
        }
    
    current_app.logger.info("음성 파일 보이스피싱 분석 완료")
    return stt_result, voicephishing_analysis
