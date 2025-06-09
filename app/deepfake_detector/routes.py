from flask import request, jsonify, current_app
from . import bp
from .services import extract_single_frame_features, analyze_temporal_features, get_final_synthesized_response
import cv2
import os
import tempfile
import requests # RAG API 호출을 위해 requests 라이브러리 임포트

# 허용되는 파일 확장자 목록 (이전과 동일)
allowed_extensions = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'wmv', 'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'}

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
    if not situation:
        return jsonify({"error": "영상/이미지와 함께 반드시 상황 설명을 입력해야 합니다."}), 400

    file_ext = uploaded_file.filename.rsplit('.', 1)[1].lower()

    try:
        # 1. 딥페이크 기술 분석
        deepfake_analysis_data = {}
        input_type = "image"
        feature_analysis_for_response = {}

        if file_ext in ['png', 'jpg', 'jpeg']:
            # 현재 로직은 동영상에 최적화 되어 있으므로, 정지 이미지는 간단히 처리
            deepfake_analysis_data = {"deepfake_probability": "판단 어려움", "reasoning": "정지 이미지는 다른 분석 기준이 필요합니다."}
        elif file_ext in ['mp4', 'mov', 'avi', 'wmv']:
            input_type = "video"
            frame_analysis_results = []
            # ... (이하 동영상 프레임 추출 및 분석 로직) ...
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                uploaded_file.save(tmp.name); video_path = tmp.name
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                os.remove(video_path); return jsonify({"error": "동영상 파일을 열 수 없습니다."}), 400
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30; total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); max_frames_to_analyze = 30
            for i in range(max_frames_to_analyze):
                frame_id = int(i * (total_frames / max_frames_to_analyze));
                if frame_id >= total_frames: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id); success, frame_np = cap.read()
                if not success: break
                _, buffer = cv2.imencode(".jpg", frame_np)
                features, _ = extract_single_frame_features(buffer.tobytes())
                if features.get("face_detected"): frame_analysis_results.append(features)
            
            cap.release(); os.remove(video_path)
            if len(frame_analysis_results) < 5: return jsonify({"error": "동영상에서 분석할 프레임이 부족합니다."}), 400
            
            # 시간적 특징 계산
            temporal_features = analyze_temporal_features(frame_analysis_results)
            # 딥페이크 판단은 이제 LLM에게 모든 데이터를 주고 맡김
            deepfake_analysis_data = {"temporal_analysis": temporal_features}
            feature_analysis_for_response = {"bounding_box": frame_analysis_results[0].get("bounding_box")}
        
        # 2. RAG 시스템에 내부 API 요청
        rag_search_results = {"results": []} # 기본값
        try:
            rag_api_url = "http://localhost:8000/search"
            rag_response = requests.post(rag_api_url, json={"query": situation, "top_k": 3})
            rag_response.raise_for_status()
            rag_search_results = rag_response.json()
        except requests.exceptions.RequestException as e:
            current_app.logger.warning(f"RAG API 호출 실패: {e}")
            rag_search_results = {"error": "유사 사례 검색에 실패했습니다.", "details": str(e)}

        # 3. 두 결과를 합쳐 최종 답변 생성
        final_response_data, final_status_code = get_final_synthesized_response(
            deepfake_analysis_data=deepfake_analysis_data,
            rag_search_results=rag_search_results,
            user_situation=situation
        )
        
        return jsonify({
            "feature_analysis": feature_analysis_for_response,
            "llm_judgment": final_response_data,
            "analyzed_input_type": input_type
        }), final_status_code

    except Exception as e:
        current_app.logger.critical(f"미디어 분석 API 처리 중 심각한 예외 발생: {e}", exc_info=True)
        return jsonify({"error": f"서버 내부 오류로 미디어 분석에 실패했습니다."}), 500

@bp.route('/analyze_audio', methods=['POST'])
def analyze_audio_endpoint():
    """음성 파일 분석 엔드포인트"""
    current_app.logger.info(f"음성 분석 요청 수신: {request.method} {request.path}")

    if 'audio' not in request.files:
        current_app.logger.warning("API 요청에 'audio' 파일 파트가 누락되었습니다.")
        return jsonify({'error': "업로드된 파일('audio' key)이 없습니다."}), 400

    uploaded_file = request.files['audio']
    filename = uploaded_file.filename

    if filename == '':
        current_app.logger.warning("빈 파일 이름으로 파일 업로드가 시도되었습니다.")
        return jsonify({"error": "업로드된 파일의 이름이 비어있습니다."}), 400

    # 음성 파일 확장자 체크
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    audio_extensions = {'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'}
    
    if file_ext not in audio_extensions:
        current_app.logger.warning(f"허용되지 않는 음성 파일 형식 업로드 시도: {filename}")
        return jsonify({"error": f"허용되지 않는 음성 파일 형식입니다. ({', '.join(audio_extensions)} 확장자만 가능)"}), 400

    situation = request.form.get('situation', "")
    
    try:
        current_app.logger.info(f"음성 파일 '{filename}' (크기: {uploaded_file.content_length}B) 분석 시작.")
        
        # 음성 파일을 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_audio_file:
            uploaded_file.save(tmp_audio_file.name)
            audio_path = tmp_audio_file.name
        
        # STT 서비스 호출
        from .services import analyze_audio_for_voicephishing
        stt_result, llm_analysis = analyze_audio_for_voicephishing(audio_path, situation)
        
        # 임시 파일 삭제
        os.remove(audio_path)
        
        current_app.logger.info("음성 파일 분석 완료.")
        
        final_response = {
            "stt_result": stt_result,
            "llm_judgment": llm_analysis,
            "analyzed_input_type": "audio"
        }
        
        return jsonify(final_response), 200

    except Exception as e:
        current_app.logger.critical(f"음성 분석 API 처리 중 심각한 예외 발생: {e}", exc_info=True)
        # 임시 파일이 있다면 삭제
        if 'audio_path' in locals():
            try:
                os.remove(audio_path)
            except:
                pass
        return jsonify({"error": f"서버 내부 오류로 음성 분석에 실패했습니다."}), 500
