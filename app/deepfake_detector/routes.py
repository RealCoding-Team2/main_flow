from flask import request, jsonify, current_app
from . import bp # 현재 패키지(__init__.py)에서 정의한 블루프린트 객체(bp)를 임포트
from .services import extract_deepfake_features # 서비스 함수들 임포트

# 허용되는 확장자 종류
allowed_extensions = {'png', 'jpg', 'jpeg'}

# /api/deepfake/analyze_image 경로로 POST 요청을 처리하는 함수
@bp.route('/analyze_image', methods=['POST'])
def analyze_image_endpoint():
    uploaded_file = request.files['image']  # 업로드된 파일 객체 가져오기
    filename = uploaded_file.filename       # 파일 이름 가져오기
    
    if 'image' not in request.files: # script.js에서 formData.append('image', ...) 로 보냄
        return jsonify({'error': '업로드된 파일이 없습니다.'}), 400
    
    if filename == '':
        current_app.logger.warning("빈 파일 이름으로 파일 업로드가 시도되었습니다.")
        return jsonify({"error": "업로드된 파일의 이름이 비어있습니다."}), 400
    
    # 현재 요청에 대한 정보 로깅
    current_app.logger.info(f"딥페이크 분석 요청 수신: {request.method} {request.path}")

    # 허용되지 않는 확장자인 경우 ( 검사결과가 False일 경우 문제 있음!! )
    if check_file_extension(filename) == False:
        current_app.logger.warning(f"허용되지 않는 파일 형식 업로드 시도: {filename}")
        return jsonify({"error": f"허용되지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 확장자만 가능)"}), 400

    try:
        # 업로드된 파일의 내용을 바이트 스트림으로 읽기
        image_data_bytes = uploaded_file.read()
        current_app.logger.info(f"파일 '{filename}' (크기: {len(image_data_bytes)} 바이트) 읽기 완료.")

        # 1. 이미지 특징 추출 서비스 호출
        extracted_features, feature_message = extract_deepfake_features(image_data_bytes)
        current_app.logger.debug(f"이미지 특징 추출 서비스 결과 메시지: {feature_message}") # 특징 데이터는 길 수 있으므로 로깅 주의
        
        return jsonify({
            "feature_analysis": "아직 개발 중"
        }), 200

    except Exception as e: # 그 외 예상치 못한 모든 예외 처리
        current_app.logger.critical(f"이미지 분석 API 엔드포인트 처리 중 심각한 예외 발생: {e}", exc_info=True) # 예외 정보 전체 로깅
        # 사용자에게는 일반적인 서버 오류 메시지 반환
        return jsonify({"error": f"서버 내부 오류로 이미지 분석에 실패했습니다. 잠시 후 다시 시도해주세요."}), 500
    
def check_file_extension(filename):
    # 파일 이름에서 확장자 추출 (소문자로 변환)
    # werkzeug.utils.secure_filename을 사용하여 파일 이름을 안전하게 처리하는 것이 좋지만, 여기서는 간단히 처리
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    print(file_ext)
    if file_ext not in allowed_extensions: # 허용되지 않는 확장자인 경우
        current_app.logger.warning(f"허용되지 않는 파일 형식 업로드 시도: {filename}")
        return False
    
    # 위 조건 검사에서 통과했으면 True
    return True