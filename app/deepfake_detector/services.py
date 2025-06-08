import cv2
import dlib
import numpy as np
from PIL import Image
import io
import json
import os
from flask import current_app
from app.llm_integration.llm import LLMRequester

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

# 눈 깜빡임 분석
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

# --- ✨새로운 기능 함수✨ ---
def _analyze_mouth_opening_from_landmarks(landmarks):
    if landmarks is None:
        return {"mouth_opening_status": "얼굴 랜드마크 데이터 없음"}
    try:
        # 입의 세로, 가로 길이를 계산하기 위한 랜드마크 인덱스 (dlib 68점 기준)
        # 입술의 안쪽 경계를 사용 (60-67)
        v_A = np.linalg.norm(np.array([landmarks.part(62).x, landmarks.part(62).y]) - np.array([landmarks.part(66).x, landmarks.part(66).y]))
        h_B = np.linalg.norm(np.array([landmarks.part(60).x, landmarks.part(60).y]) - np.array([landmarks.part(64).x, landmarks.part(64).y]))

        if h_B == 0: return {"mouth_aspect_ratio": 0.0, "mouth_opening_status": "입 너비 0, 계산 불가"}
        
        mar = v_A / h_B # Mouth Aspect Ratio 계산
        
        status = "입 다묾"
        if mar > 0.5: status = "입 크게 벌림"
        elif mar > 0.2: status = "입 약간 벌림"

        return {
            "mouth_aspect_ratio": round(float(mar), 4),
            "mouth_opening_status": status
        }
    except Exception as e:
        current_app.logger.error(f"입 움직임 분석 중 오류 발생: {e}", exc_info=True)
        return {"mouth_opening_status": f"입 움직임 분석 중 오류: {str(e)}"}


# 이미지 바이트(단일 프레임)로부터 딥페이크 관련 특징 추출
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
    
    # 각 분석 함수 호출
    eye_blinking_features = analyze_eye_blinking_from_landmarks(main_face_landmarks)
    facial_consistency_features = analyze_facial_consistency_from_landmarks(main_face_landmarks)
    mouth_opening_features = _analyze_mouth_opening_from_landmarks(main_face_landmarks) # ✨새로운 기능 호출✨

    # 최종 데이터 구성
    face_rect = detected_faces[0]
    extracted_features_data = {
        "face_detected": True,
        "bounding_box": {
            "x": face_rect.left(), "y": face_rect.top(),
            "w": face_rect.width(), "h": face_rect.height()
        },
        "eye_blinking_analysis": eye_blinking_features,
        "facial_consistency_analysis": facial_consistency_features,
        "mouth_opening_analysis": mouth_opening_features, # ✨새로운 분석 결과 추가✨
    }
    current_app.logger.info("단일 프레임으로부터 특징 추출 성공.")
    return extracted_features_data, "특징 추출 성공"

# LLM에게 최종 판단 요청 (변경 없음)
def get_llm_deepfake_judgment(analysis_input_data, situation_description="", input_type="image"):
    try:
        llm_model_to_use = os.getenv("DEFAULT_MODEL") or "gpt-3.5-turbo"
        llm_requester = LLMRequester(model=llm_model_to_use)
        current_app.logger.info(f"OpenAI LLM ({llm_model_to_use}) 객체 생성 완료.")
    except ValueError as ve:
        current_app.logger.error(f"LLMRequester 초기화 오류 (아마도 API 키 부재): {ve}")
        return {
            "deepfake_probability": "판단 불가 (LLM API 키 설정 오류)",
            "reasoning": f"내부 LLM 서비스 초기화 중 오류가 발생했습니다: {str(ve)}",
            "recommendations_for_user": "관리자에게 문의하여 LLM 설정을 확인해주세요."
        }, 500
    except Exception as e_init:
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
        llm_model_to_use = os.getenv("DEFAULT_MODEL")
        llm_requester = LLMRequester(model=llm_model_to_use)

        llm_text_response = llm_requester.send_message(
            message=user_message_content,
            system_prompt=system_prompt_for_openai
        )
        current_app.logger.debug(f"OpenAI LLM 원시 응답: {llm_text_response}")

        try:
            parsed_llm_response = json.loads(llm_text_response)
            if not all(key in parsed_llm_response for key in ["deepfake_probability", "reasoning", "recommendations_for_user"]):
                current_app.logger.warning(f"LLM 응답이 예상된 JSON 형식을 완전히 따르지 않음: {parsed_llm_response}")
            return parsed_llm_response, 200
        except json.JSONDecodeError as e:
            current_app.logger.error(f"OpenAI LLM 응답 JSON 파싱 오류: {e} - 응답 내용: {llm_text_response}")
            return {"error": "LLM 응답 형식 오류 (JSON 파싱 실패)", "raw_response": llm_text_response}, 500
        except TypeError as te:
             current_app.logger.error(f"OpenAI LLM 응답 타입 오류 (이미 객체일 수 있음): {te} - 응답: {llm_text_response}")
             if isinstance(llm_text_response, dict):
                 return llm_text_response, 200
             return {"error": "LLM 응답 타입 오류", "raw_response": str(llm_text_response)}, 500

    except ValueError as ve:
        current_app.logger.error(f"LLMRequester 초기화 오류: {ve}")
        return {"error": f"LLM 서비스 설정 오류: {ve}"}, 500
    except Exception as e:
        current_app.logger.critical(f"LLM 판단 중 예외 발생: {e}", exc_info=True)
        return {"error": f"LLM 판단 중 오류 발생: {str(e)}"}, 500