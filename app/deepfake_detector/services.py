import cv2
import dlib
import numpy as np
from PIL import Image
import io
import json
import os
from flask import current_app
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
        llm_model_to_use = os.getenv("DEFAULT_MODEL") or "gpt-4o-mini"
        llm_requester = LLMRequester(model=llm_model_to_use)
    except Exception as e:
        return {"error": f"LLM 서비스 초기화 오류: {e}"}, 500

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
        llm_text_response = llm_requester.send_message(message=user_message_for_master_ai, system_prompt=system_prompt_for_master_ai)
        return json.loads(llm_text_response), 200
    except Exception as e:
        current_app.logger.error(f"최종 답변 생성 중 오류 발생: {e}")
        return {"error": f"최종 답변 생성 중 오류 발생: {e}"}, 500