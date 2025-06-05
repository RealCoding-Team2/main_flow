
import sys
import os
from flask import Flask  
from flask_cors import CORS
import logging
from config import Config 

'''
1. 모든 딥페이크탐지 API는 /api/deepfake/ 로 시작

'''

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class) # Config 클래스의 설정 적용

    # Config 클래스의 init_app 메서드를 호출하여 추가 설정 (Dlib 경로, Gemini URL 등)
    config_class.init_app(app)

    # 웹 챗봇 프론트엔드와 통신하기 위한 CORS 설정
    # 개발 중에는 모든 출처(*)를 허용하지만, 실제 배포 시에는 프론트엔드 도메인만 명시하는 것이 안전합니다.
    CORS(app, resources={r"/api/*": {"origins": "*"}}) # /api/ 로 시작하는 모든 경로에 CORS 적용

    # 로깅 설정 (앱의 동작 상태를 기록)
    log_level = logging.DEBUG if app.config.get('FLASK_DEBUG') else logging.INFO
    if not app.logger.handlers: # 핸들러 중복 추가 방지
        stream_handler = logging.StreamHandler()
        # 로그 포맷 상세하게 변경
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        stream_handler.setFormatter(formatter)
        app.logger.addHandler(stream_handler)
        
    app.logger.setLevel(log_level)

    app.logger.info('Flask 애플리케이션이 성공적으로 생성되고 설정되었습니다.')
    app.logger.debug(f"현재 디버그 모드: {app.config.get('FLASK_DEBUG')}")
    app.logger.info(f"Dlib 모델 파일 경로: {app.config.get('DLIB_LANDMARK_MODEL_PATH')}")
    
    # # LLM API키를 불러오고 일부분을 보여줍니다.
    # api_url = app.config.get('GEMINI_API_URL')
    # logging_key(app, api_url)

     # --- LLM API 키 설정 상태 로깅 (OpenAI 기준으로 변경) ---
    if app.config.get('OPENAI_API_KEY_SET'):
        # yongs3님의 llm.py에서 사용하는 DEFAULT_MODEL 환경 변수를 가져와서 로깅할 수도 있습니다.
        default_model_from_env = os.getenv("DEFAULT_MODEL", "정보 없음") # llm.py가 .env를 직접 쓰므로 여기서도 os.getenv 사용
        app.logger.info(f"OpenAI API 키가 설정되었습니다. (기본 모델: {default_model_from_env})")
        # OPENAI_BASE_URL 로깅은 필요하다면 추가
        # base_url_from_env = os.getenv("OPENAI_BASE_URL")
        # if base_url_from_env:
        #    app.logger.info(f"OpenAI Base URL: {base_url_from_env}")
    else:
        app.logger.warning("OpenAI API 키가 설정되지 않았습니다. LLM 기능이 제한될 수 있습니다.")
    # --- 기존 logging_key 함수 호출 및 Gemini 관련 로그는 제거 ---
    # api_url = app.config.get('GEMINI_API_URL') # 삭제 또는 주석
    # logging_key(app, api_url) # 삭제 또는 주석
    
    # 딥페이크 탐지 기능을 위한 블루프린트 등록
    from .deepfake_detector import bp as deepfake_bp
    
    app.register_blueprint(deepfake_bp, url_prefix='/api/deepfake') 
    app.logger.info("Deepfake detector 블루프린트가 '/api/deepfake' 경로로 등록되었습니다.")

    # 서버 상태 확인용 간단한 경로
    @app.route('/health')
    def health_check():
        app.logger.debug("Health check 엔드포인트가 호출되었습니다.")
        return "딥페이크 탐지 서버가 정상 작동 중입니다!", 200

    return app

# def logging_key(app, api_url):
#     if api_url:
#         key_param_index = api_url.find('key=')
#         log_api_url = api_url[:key_param_index + 4] + '*****' if key_param_index != -1 else api_url
        
#         # API 키의 일부만 로깅
#         app.logger.info(f"Gemini API URL (키 마스킹됨): {log_api_url}")
#     else:
#         app.logger.warning("Gemini API URL이 설정되지 않았습니다.")
    