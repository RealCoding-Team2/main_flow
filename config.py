import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print(f"경고: .env 파일을 찾을 수 없습니다. 경로: {dotenv_path}. 환경 변수가 직접 설정되었는지 확인하세요.")

class Config: # <--- 이 클래스 이름이 정확해야 합니다!
    FLASK_DEBUG    = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    GEMINI_MODEL_NAME   = "gemini-1.5-flash-latest"
    GEMINI_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    DLIB_LANDMARK_MODEL_PATH = None

    @staticmethod
    def get_gemini_api_url():
        if not Config.GEMINI_API_KEY or not Config.GEMINI_MODEL_NAME:
            return None
        
        return f"{Config.GEMINI_API_URL_BASE}/{Config.GEMINI_MODEL_NAME}:generateContent?key={Config.GEMINI_API_KEY}"

    @staticmethod
    def init_app(app):
        project_root = os.path.dirname(os.path.abspath(__file__))
        app.config['DLIB_LANDMARK_MODEL_PATH'] = os.path.join(project_root, 'app', 'models', 'shape_predictor_68_face_landmarks.dat')
        app.config['GEMINI_API_URL'] = Config.get_gemini_api_url()

        if not app.config.get('GEMINI_API_KEY'):
            if hasattr(app, 'logger'): # logger가 초기화되었는지 확인
                app.logger.warning("GEMINI_API_KEY가 설정되지 않았습니다. LLM 연동이 안 될 수 있습니다.")
            else:
                print("경고 (logger 없음): GEMINI_API_KEY가 설정되지 않았습니다.")
        if not os.path.exists(app.config['DLIB_LANDMARK_MODEL_PATH']):
            if hasattr(app, 'logger'):
                app.logger.error(f"Dlib 모델 파일을 찾을 수 없습니다: {app.config['DLIB_LANDMARK_MODEL_PATH']}")
            else:
                print(f"에러 (logger 없음): Dlib 모델 파일을 찾을 수 없습니다: {app.config['DLIB_LANDMARK_MODEL_PATH']}")