
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
        model_path = current_app.config.get('DLIB_LANDMARK_MODEL_PATH');
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
            A = np.linalg.norm(eye_landmarks_pts[1] - eye_landmarks_pts[5]); B = np.linalg.norm(eye_landmarks_pts[2] - eye_landmarks_pts[4]); C = np.linalg.norm(eye_landmarks_pts[0] - eye_landmarks_pts[3])
            if C == 0: return 0.0
            return (A + B) / (2.0 * C)
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]); right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_ear, right_ear = eye_aspect_ratio(left_eye_pts), eye_aspect_ratio(right_eye_pts); avg_ear = (left_ear + right_ear) / 2.0; status = "눈 깜빡임 보통 (EAR 일반 범위)"
        if avg_ear < 0.18: status = "눈 감았을 가능성 매우 높음 (EAR 매우 낮음)"
        elif avg_ear < 0.22: status = "눈 약간 감았거나 깜빡이는 중일 가능성 (EAR 낮음)"
        elif avg_ear > 0.28: status = "눈 떴을 가능성 (EAR 보통)"
        return {"left_eye_ear": round(float(left_ear), 4), "right_eye_ear": round(float(right_ear), 4), "average_ear": round(float(avg_ear), 4), "eye_blinking_status": status}
    except Exception as e: return {"eye_blinking_status": f"눈 깜빡임 분석 중 오류: {str(e)}"}
def analyze_facial_consistency_from_landmarks(landmarks):
    if landmarks is None: return {"facial_consistency_status": "얼굴 랜드마크 데이터 없음"}
    try:
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y); chin_tip = (landmarks.part(8).x, landmarks.part(8).y); left_eye_inner_corner = (landmarks.part(39).x, landmarks.part(39).y); right_eye_inner_corner = (landmarks.part(42).x, landmarks.part(42).y)
        nose_to_chin_distance = np.linalg.norm(np.array(nose_tip) - np.array(chin_tip)); inter_pupillary_distance_approx = np.linalg.norm(np.array(left_eye_inner_corner) - np.array(right_eye_inner_corner)); face_rect = landmarks.rect
        nose_symmetry_offset = abs(nose_tip[0] - (face_rect.left() + face_rect.width() / 2))
        return {"nose_to_chin_distance": round(float(nose_to_chin_distance), 2), "inter_pupillary_distance_approx": round(float(inter_pupillary_distance_approx), 2), "nose_symmetry_offset": round(float(nose_symmetry_offset), 2), "facial_consistency_status": "분석 완료"}
    except Exception as e: return {"facial_consistency_status": f"얼굴 일관성 분석 중 오류: {str(e)}"}
def _analyze_mouth_opening_from_landmarks(landmarks):
    if landmarks is None: return {"mouth_opening_status": "얼굴 랜드마크 데이터 없음"}
    try:
        v_A = np.linalg.norm(np.array([landmarks.part(62).x, landmarks.part(62).y]) - np.array([landmarks.part(66).x, landmarks.part(66).y])); h_B = np.linalg.norm(np.array([landmarks.part(60).x, landmarks.part(60).y]) - np.array([landmarks.part(64).x, landmarks.part(64).y]))
        if h_B == 0: return {"mouth_aspect_ratio": 0.0, "mouth_opening_status": "입 너비 0, 계산 불가"}
        mar = v_A / h_B; status = "입 다묾"
        if mar > 0.5: status = "입 크게 벌림"
        elif mar > 0.2: status = "입 약간 벌림"
        return {"mouth_aspect_ratio": round(float(mar), 4), "mouth_opening_status": status}
    except Exception as e: return {"mouth_opening_status": f"입 움직임 분석 중 오류: {str(e)}"}
def _analyze_head_pose_from_landmarks(landmarks, image_shape):
    if landmarks is None: return {"head_pose_status": "얼굴 랜드마크 데이터 없음"}
    try:
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        image_points = np.array([(landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(8).x,  landmarks.part(8).y), (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y), (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)], dtype="double")
        focal_length = image_shape[1]; center = (image_shape[1]/2, image_shape[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"); dist_coeffs = np.zeros((4, 1))
        (_, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector); projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()[:3]
        return {"pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2), "head_pose_status": "분석 완료"}
    except Exception as e: return {"head_pose_status": f"머리 자세 추정 중 오류: {str(e)}"}
def _stabilize_angle_series(angle_series, smoothing_window=3):
    if len(angle_series) < smoothing_window: return np.array(angle_series)
    unwrapped_angles = np.unwrap(np.deg2rad(angle_series)); unwrapped_degrees = np.rad2deg(unwrapped_angles)
    smoothed_angles = pd.Series(unwrapped_degrees).rolling(window=smoothing_window, min_periods=1, center=True).mean().to_numpy()
    return smoothed_angles
def analyze_temporal_features(frame_features_list):
    if not frame_features_list or len(frame_features_list) < 5: return {"temporal_analysis_status": "분석할 프레임 부족"}
    yaws = [f.get('head_pose_analysis', {}).get('yaw', 0) for f in frame_features_list]; pitches = [f.get('head_pose_analysis', {}).get('pitch', 0) for f in frame_features_list]; rolls = [f.get('head_pose_analysis', {}).get('roll', 0) for f in frame_features_list]
    yaws_stable = _stabilize_angle_series(yaws); pitches_stable = _stabilize_angle_series(pitches); rolls_stable = _stabilize_angle_series(rolls)
    yaw_velocities = np.diff(yaws_stable); pitch_velocities = np.diff(pitches_stable); roll_velocities = np.diff(rolls_stable)
    return {
        "position_jitter_stable": { "yaw_std": round(np.std(yaws_stable), 2), "pitch_std": round(np.std(pitches_stable), 2), "roll_std": round(np.std(rolls_stable), 2), },
        "velocity_jitter_stable": { "yaw_velocity_std": round(np.std(yaw_velocities), 2) if len(yaw_velocities) > 0 else 0, "pitch_velocity_std": round(np.std(pitch_velocities), 2) if len(pitch_velocities) > 0 else 0, "roll_velocity_std": round(np.std(roll_velocities), 2) if len(roll_velocities) > 0 else 0, },
        "range_of_motion_stable": { "yaw_range": round(np.max(yaws_stable) - np.min(yaws_stable), 2) if yaws_stable.size > 0 else 0, "pitch_range": round(np.max(pitches_stable) - np.min(pitches_stable), 2) if pitches_stable.size > 0 else 0, "roll_range": round(np.max(rolls_stable) - np.min(rolls_stable), 2) if rolls_stable.size > 0 else 0, },
        "temporal_analysis_status": "분석 완료"
    }
def extract_single_frame_features(image_bytes):
    detector, predictor = get_dlib_objects_with_caching();
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
        "facial_consistency_analysis": analyze_facial_consistency_from_landmarks(main_face_landmarks),
        "mouth_opening_analysis": _analyze_mouth_opening_from_landmarks(main_face_landmarks),
        "head_pose_analysis": _analyze_head_pose_from_landmarks(main_face_landmarks, image_shape),
    }, "특징 추출 성공"

def get_llm_deepfake_judgment(analysis_input_data, situation_description="", input_type="image"):
    try:
        llm_model_to_use = os.getenv("DEFAULT_MODEL") or "gpt-4o-mini"
        llm_requester = LLMRequester(model=llm_model_to_use)
    except Exception as e:
        return {"error": f"LLM 서비스 초기화 오류: {e}"}, 500

    # --- ✨최종 버전 시스템 프롬프트✨ ---
    system_prompt_for_data_scientist = """당신은 영상 분석을 통해 딥페이크를 탐지하는 최고 수준의 AI 데이터 과학자입니다. 제공된 정량적 데이터를 기반으로, 아래의 엄격한 규칙에 따라 논리적으로 추론하고 JSON 형식으로만 답변해야 합니다.

[핵심 분석 데이터]
- `velocity_jitter_stable`: 머리 움직임 속도의 '불규칙성' 또는 '끊김' 정도를 나타냅니다. 이 지표가 딥페이크 판단의 가장 결정적인 증거입니다.
- 다른 모든 데이터(`position_jitter`, `range_of_motion` 등)는 보조적인 참고 자료로만 활용하세요.

[딥페이크 확률 판단 규칙]
1.  `velocity_jitter_stable`의 `yaw_velocity_std`, `pitch_velocity_std`, `roll_velocity_std` 세 값 중 **두 개 이상이 4.0을 초과**하면, 확률을 **"매우 높음"**으로 판단합니다.
2.  세 값 중 **하나라도 4.0을 초과**하면, 확률을 **"높음"**으로 판단합니다.
3.  세 값 중 **하나라도 2.0을 초과**하면, 확률을 **"중간"**으로 판단합니다.
4.  세 값이 **모두 1.0 미만**이면, 확률을 **"매우 낮음"**으로 판단합니다.
5.  그 외의 모든 경우는 확률을 **"낮음"**으로 판단합니다.

[신뢰도(confidence_score) 설정 규칙]
- "매우 높음" 또는 "매우 낮음"으로 판단한 경우: 판단 근거가 명확하므로 신뢰도를 **0.9 이상**으로 높게 설정합니다.
- "높음" 또는 "낮음"으로 판단한 경우: 판단 근거가 어느 정도 있으므로 신뢰도를 **0.7 ~ 0.85 사이**로 설정합니다.
- "중간"으로 판단한 경우: 판단 근거가 애매하므로 신뢰도를 **0.5 ~ 0.65 사이**로 설정합니다.

[필수 JSON 응답 형식]
{
  "deepfake_probability": "string (위 규칙에 따라 판단된 값)",
  "confidence_score": "float (위 규칙에 따라 설정된 0.0 ~ 1.0 값)",
  "reasoning": "string (어떤 지표(들)가 규칙을 만족하여 이런 결론을 내렸는지 설명)",
  "recommendations_for_user": "string (사용자를 위한 일반적인 조언)"
}

[답변 스타일 규칙]
`reasoning` 작성 시, 기술 용어 대신 '머리 움직임의 속도 변화가 여러 방향에서 매우 불규칙하고 부자연스러운 점' 또는 '전반적인 움직임이 부드러웠던 점'과 같이 쉬운 언어로 설명해야 합니다.
"""
    
    input_type_description = "정지된 이미지" if input_type == "image" else "동영상"
    user_message_content = f"""
[분석 대상 정보]
- 입력 타입: {input_type_description}
- 분석 데이터 요약: {json.dumps(analysis_input_data, indent=2, ensure_ascii=False)}
- 사용자 상황 설명: {situation_description if situation_description else "제공되지 않음"}

위 정보를 바탕으로, 규칙에 맞게 JSON 형식으로만 딥페이크 분석 결과를 제공해주세요.
"""

    try:
        llm_text_response = llm_requester.send_message(message=user_message_content, system_prompt=system_prompt_for_data_scientist)
        return json.loads(llm_text_response), 200
    except Exception as e:
        current_app.logger.error(f"LLM 판단 중 오류 발생: {e} - 응답 내용: {llm_text_response if 'llm_text_response' in locals() else 'N/A'}")
        return {"error": f"LLM 판단 중 오류 발생: {e}"}, 500