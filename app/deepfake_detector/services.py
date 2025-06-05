
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
# def get_llm_deepfake_judgment(analysis_input_data, situation_description="", input_type="image"):
#     """
#     analysis_input_data:
#         - 이미지가 하나일 경우: extract_single_frame_features의 반환값 (dict)
#         - 동영상 프레임이 여러 개일 경우: 각 프레임의 분석 결과 리스트 (list of dicts)
#     input_type: "image" 또는 "video"
#     """
#     if not current_app.config.get('GEMINI_API_KEY'):
#         current_app.logger.warning("LLM 판단을 건너뜁니다: GEMINI_API_KEY 설정 없음.")
#         return {
#             "deepfake_probability": "판단 불가 (LLM 설정 없음)",
#             "reasoning": "내부 LLM 서비스가 설정되지 않아 자동 판단을 수행할 수 없습니다.",
#             "recommendations_for_user": "추출된 특징 정보를 참고하여 직접 판단하세요."
#         }, 200

#     # 프롬프트 구성 (입력 타입에 따라 다르게 구성 가능)
#     input_type_description = "정지된 이미지" if input_type == "image" else "동영상"
    
#     # 동영상의 경우, 여러 프레임의 정보를 요약하거나 대표적인 정보를 전달
#     if input_type == "video" and isinstance(analysis_input_data, list):
#         # 간단한 요약 예시: 얼굴 감지 여부, 평균 EAR 등
#         num_frames_analyzed = len(analysis_input_data)
#         frames_with_face = [f for f in analysis_input_data if f.get("face_detected")]
#         num_frames_with_face = len(frames_with_face)
        
#         ear_values = [f["eye_blinking_analysis"]["average_ear"] 
#                       for f in frames_with_face 
#                       if f.get("eye_blinking_analysis") and "average_ear" in f["eye_blinking_analysis"]]
#         avg_ear_overall = sum(ear_values) / len(ear_values) if ear_values else "N/A"

#         feature_summary_for_llm = {
#             "input_type": input_type_description,
#             "total_frames_analyzed": num_frames_analyzed,
#             "frames_with_face_detected": num_frames_with_face,
#             "average_ear_across_frames_with_face": round(avg_ear_overall, 4) if isinstance(avg_ear_overall, float) else avg_ear_overall,
#             # 필요시 다른 프레임별 주요 특징 요약 추가
#             # "first_frame_analysis": analysis_input_data[0] if analysis_input_data else None # 예시
#         }
#         current_app.logger.info(f"동영상 분석 요약 정보 (LLM 전달용): {feature_summary_for_llm}")
#     else: # 단일 이미지의 경우
#         feature_summary_for_llm = analysis_input_data
#         current_app.logger.info(f"단일 이미지 분석 정보 (LLM 전달용): {feature_summary_for_llm}")


#     prompt = f"""당신은 고도로 숙련된 딥페이크 탐지 전문가 AI입니다.
# 제공된 입력은 **{input_type_description}**입니다.
# 제공된 분석 데이터와 사용자 상황 설명을 면밀히 분석하여, 해당 콘텐츠가 딥페이크일 가능성을 평가하고, 그 판단에 대한 상세한 근거와 신뢰도, 사용자 조언을 포함한 JSON 형식의 보고서를 작성해주세요.
# 응답은 반드시 아래 명시된 JSON 구조만을 사용해야 하며, 다른 어떤 텍스트도 포함해서는 안 됩니다.

# [분석 데이터 요약]
# {json.dumps(feature_summary_for_llm, indent=2, ensure_ascii=False)}

# [사용자 상황 설명]
# {situation_description if situation_description else "제공되지 않음"}

# [판단 가이드라인]
# - 동영상의 경우, 프레임 간의 일관성 변화도 중요한 단서가 될 수 있습니다. (현재 요약 정보에는 포함되지 않았으나, 일반적인 고려 사항)
# - 정지 이미지의 경우, 눈 깜빡임(EAR)은 단일 값이며 동적인 패턴을 의미하지 않습니다.

# [필수 JSON 응답 형식 - 이 형식과 키 이름을 정확히 따르세요]
# {{
#   "deepfake_probability": "string (다음 중 하나: 매우 높음, 높음, 중간, 낮음, 매우 낮음, 판단 어려움)",
#   "confidence_score": "float (0.0에서 1.0 사이의 판단 신뢰도 점수)",
#   "reasoning": "string (판단 근거를 상세히 설명. 어떤 특징이 의심을 증폭시키거나 감소시키는지 명시)",
#   "key_indicators": ["string (딥페이크 판단에 결정적이었던 주요 지표들 목록)"],
#   "recommendations_for_user": "string (사용자를 위한 구체적인 다음 단계 조언이나 주의사항)"
# }}
# """
#     current_app.logger.info("Gemini LLM을 통한 딥페이크 판단 요청을 시작합니다...")
#     return call_gemini_api(prompt)

# phishing_detection_project/main_flow/app/deepfake_detector/services.py

# ... (get_dlib_objects_with_caching, analyze_eye_blinking_from_landmarks,
#      analyze_facial_consistency_from_landmarks, extract_single_frame_features 함수는 동일하게 유지) ...

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
        llm_model_to_use = os.getenv("DEFAULT_MODEL") or "gpt-3.5-turbo" # 예시 모델, yongs3님 설정 확인 필요
        llm_requester = LLMRequester(model=llm_model_to_use)
        current_app.logger.info(f"OpenAI LLM ({llm_model_to_use})을 통한 딥페이크 판단 요청 시작...")

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