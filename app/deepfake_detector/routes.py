from flask import request, jsonify, current_app
from . import bp
from .services import extract_single_frame_features, get_llm_deepfake_judgment, analyze_temporal_features
import cv2
import os
import tempfile

allowed_extensions = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'wmv'}

def check_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@bp.route('/analyze_image', methods=['POST'])
def analyze_media_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': "업로드된 파일('image' key)이 없습니다."}), 400

    uploaded_file = request.files['image']
    if not uploaded_file.filename or not check_file_extension(uploaded_file.filename):
        return jsonify({"error": "파일이 없거나 허용되지 않는 형식입니다."}), 400

    situation = request.form.get('situation', "")
    file_ext = uploaded_file.filename.rsplit('.', 1)[1].lower()

    try:
        if file_ext in ['png', 'jpg', 'jpeg']:
            input_type = "image"
            image_data_bytes = uploaded_file.read()
            analysis_input_data, _ = extract_single_frame_features(image_data_bytes)
            feature_analysis_for_response = {"bounding_box": analysis_input_data.get("bounding_box")}
        
        elif file_ext in ['mp4', 'mov', 'avi', 'wmv']:
            input_type = "video"
            frame_analysis_results = []
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_video_file:
                uploaded_file.save(tmp_video_file.name)
                video_path_for_cv2 = tmp_video_file.name
            
            cap = cv2.VideoCapture(video_path_for_cv2)
            if not cap.isOpened():
                os.remove(video_path_for_cv2)
                return jsonify({"error": "동영상 파일을 열 수 없습니다."}), 400

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames_to_analyze = 30 

            for i in range(max_frames_to_analyze):
                frame_id = int(i * (total_frames / max_frames_to_analyze))
                if frame_id >= total_frames: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, frame_np = cap.read()
                if not success: break

                _, buffer = cv2.imencode(".jpg", frame_np)
                features, _ = extract_single_frame_features(buffer.tobytes())
                if features.get("face_detected"):
                    frame_analysis_results.append(features)

            cap.release()
            os.remove(video_path_for_cv2)

            if len(frame_analysis_results) < 5:
                return jsonify({"error": "동영상에서 분석에 필요한 충분한 얼굴 정보를 찾지 못했습니다."}), 400
            
            # 시간적 특징 분석
            temporal_features = analyze_temporal_features(frame_analysis_results)
            
            # LLM에 보낼 최종 데이터 (개별 프레임 정보는 제외하고 시간적 특징만 전달)
            analysis_input_data = {
                "temporal_analysis": temporal_features
            }
            feature_analysis_for_response = {"bounding_box": frame_analysis_results[0].get("bounding_box")}
        
        else:
            return jsonify({"error": "처리할 수 없는 파일 형식입니다."}), 500

        # 공통 로직: LLM 판단 요청
        llm_judgment_data, llm_status_code = get_llm_deepfake_judgment(analysis_input_data, situation, input_type)

        return jsonify({
            "feature_analysis": feature_analysis_for_response,
            "llm_judgment": llm_judgment_data,
            "analyzed_input_type": input_type
        }), llm_status_code

    except Exception as e:
        current_app.logger.critical(f"미디어 분석 API 처리 중 심각한 예외 발생: {e}", exc_info=True)
        return jsonify({"error": f"서버 내부 오류가 발생했습니다."}), 500