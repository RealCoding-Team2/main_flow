
from flask import request, jsonify, current_app
from . import bp
from .services import extract_single_frame_features, get_llm_deepfake_judgment
import cv2 # OpenCV 임포트
import os # 파일 저장을 위해
import tempfile # 임시 파일/폴더 생성을 위해
import numpy as np # Numpy 배열 사용

# 허용되는 파일 확장자 목록 (이전과 동일)
allowed_extensions = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'wmv'}

def check_file_extension(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@bp.route('/analyze_image', methods=['POST']) # 엔드포인트 이름은 유지 (이미지/동영상 모두 처리)
def analyze_media_endpoint():
    current_app.logger.info(f"미디어 분석 요청 수신: {request.method} {request.path}")

    if 'image' not in request.files: # 프론트엔드에서는 'image' 키로 파일 전송
        current_app.logger.warning("API 요청에 'image' 파일 파트가 누락되었습니다.")
        return jsonify({'error': "업로드된 파일('image' key)이 없습니다."}), 400

    uploaded_file = request.files['image']
    filename = uploaded_file.filename

    if filename == '':
        current_app.logger.warning("빈 파일 이름으로 파일 업로드가 시도되었습니다.")
        return jsonify({"error": "업로드된 파일의 이름이 비어있습니다."}), 400

    if not check_file_extension(filename):
        current_app.logger.warning(f"허용되지 않는 파일 형식 업로드 시도: {filename}")
        return jsonify({"error": f"허용되지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 확장자만 가능)"}), 400

    situation = request.form.get('situation', "")
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    analysis_input_data = None
    input_type = "image" # 기본값은 이미지

    try:
        if file_ext in ['png', 'jpg', 'jpeg']: # 이미지 파일 처리
            input_type = "image"
            image_data_bytes = uploaded_file.read()
            current_app.logger.info(f"이미지 파일 '{filename}' (크기: {len(image_data_bytes)}B) 분석 시작.")
            analysis_input_data, feature_message = extract_single_frame_features(image_data_bytes)
            current_app.logger.debug(f"이미지 특징 추출 결과 메시지: {feature_message}")

        elif file_ext in ['mp4', 'mov', 'avi', 'wmv']: # 동영상 파일 처리
            input_type = "video"
            current_app.logger.info(f"동영상 파일 '{filename}' 분석 시작.")
            
            # 동영상 프레임 추출 및 분석 (예시: 처음 5초 동안 1초에 1프레임, 최대 5프레임)
            frame_analysis_results = []
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_video_file:
                uploaded_file.save(tmp_video_file.name)
                video_path_for_cv2 = tmp_video_file.name
            
            current_app.logger.info(f"임시 동영상 파일 저장: {video_path_for_cv2}")

            cap = cv2.VideoCapture(video_path_for_cv2)
            if not cap.isOpened():
                os.remove(video_path_for_cv2) # 임시 파일 삭제
                current_app.logger.error(f"동영상 파일 '{filename}'을 열 수 없습니다.")
                return jsonify({"error": "동영상 파일을 열 수 없습니다."}), 400

            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30 # 초당 프레임 수
            max_frames_to_extract = 5 # 최대 추출 프레임 수
            seconds_to_analyze = 5    # 분석할 동영상 앞부분 시간 (초)
            frames_extracted_count = 0
            
            frame_interval = int(fps) # 1초당 1프레임 (fps 값에 따라 조절 가능)
            current_frame_num = 0

            while frames_extracted_count < max_frames_to_extract and \
                  (current_frame_num / fps) < seconds_to_analyze :
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
                success, frame_np = cap.read()
                if not success:
                    break # 프레임 읽기 실패 시 중단

                is_success_encode, buffer = cv2.imencode(".jpg", frame_np)
                if is_success_encode:
                    frame_bytes = buffer.tobytes()
                    features, msg = extract_single_frame_features(frame_bytes)
                    if features: # 유효한 특징이 추출된 경우만 추가
                        frame_analysis_results.append(features)
                    current_app.logger.debug(f"동영상 프레임 {frames_extracted_count + 1} 분석: {msg}")
                    frames_extracted_count += 1
                
                current_frame_num += frame_interval # 다음 분석할 프레임으로 이동

            cap.release()
            os.remove(video_path_for_cv2) # 임시 파일 삭제
            current_app.logger.info(f"동영상 '{filename}'에서 {frames_extracted_count}개 프레임 분석 완료.")

            if not frame_analysis_results: # 분석된 프레임이 하나도 없는 경우
                current_app.logger.warning(f"동영상 '{filename}'에서 유효한 분석 결과를 얻지 못했습니다.")
                # 얼굴 미검출과 유사한 응답 반환
                return jsonify({
                    "feature_analysis": {"face_detected": False, "error": "동영상에서 분석 가능한 얼굴을 찾지 못했습니다."},
                    "llm_judgment": {
                        "deepfake_probability": "판단 불가 (분석 가능한 얼굴 없음)",
                        "reasoning": "동영상에서 분석 가능한 얼굴 정보를 충분히 추출하지 못했습니다.",
                        "recommendations_for_user": "얼굴이 더 명확하게 나오는 다른 동영상이나 이미지를 사용해 주세요."
                    }
                }), 200
            
            analysis_input_data = frame_analysis_results # 여러 프레임의 분석 결과를 리스트로 전달

        else: # 로직상 여기까지 오면 안 되지만, 방어적으로
            current_app.logger.error(f"처리할 수 없는 파일 확장자입니다: {file_ext}")
            return jsonify({"error": "내부 서버 오류: 파일 타입 처리 실패"}), 500

        # --- 공통 로직: 특징 추출 결과 및 LLM 판단 ---
        if analysis_input_data is None or \
           (isinstance(analysis_input_data, dict) and analysis_input_data.get("error")) or \
           (isinstance(analysis_input_data, list) and not analysis_input_data) : # 리스트인데 비어있는 경우
            error_msg = "분석 데이터 생성 실패"
            if isinstance(analysis_input_data, dict) and analysis_input_data.get("error"):
                error_msg = analysis_input_data.get("error")
            current_app.logger.error(f"특징 추출/준비에 실패했습니다: {error_msg}")
            return jsonify({"error": f"특징 추출/준비 실패: {error_msg}"}), 400

        # 단일 이미지이고 얼굴 미검출 시 (동영상은 위에서 이미 처리)
        if input_type == "image" and not analysis_input_data.get("face_detected", False):
            current_app.logger.info("이미지에서 얼굴이 감지되지 않아 LLM 호출을 생략합니다.")
            return jsonify({
                "feature_analysis": analysis_input_data,
                "llm_judgment": {
                    "deepfake_probability": "판단 불가 (얼굴 미검출)",
                    "reasoning": "제공된 이미지에서 얼굴을 감지할 수 없어 딥페이크 분석을 수행할 수 없습니다.",
                    "recommendations_for_user": "얼굴이 선명하게 나온 다른 이미지를 사용해 주세요."
                }
            }), 200
        
        llm_judgment_data, llm_status_code = get_llm_deepfake_judgment(analysis_input_data, situation, input_type)
        current_app.logger.info(f"LLM 판단 서비스 완료 (HTTP 상태 코드: {llm_status_code}).")

        final_api_response = {
            # 동영상 분석 시 analysis_input_data는 프레임별 결과 리스트가 될 수 있음.
            # 프론트엔드에서 이를 어떻게 보여줄지 고려 필요. 여기서는 그대로 전달.
            "feature_analysis_summary" if input_type == "video" else "feature_analysis": analysis_input_data,
            "llm_judgment": llm_judgment_data,
            "analyzed_input_type": input_type
        }
        return jsonify(final_api_response), llm_status_code

    except Exception as e:
        current_app.logger.critical(f"미디어 분석 API 처리 중 심각한 예외 발생: {e}", exc_info=True)
        return jsonify({"error": f"서버 내부 오류로 미디어 분석에 실패했습니다."}), 500