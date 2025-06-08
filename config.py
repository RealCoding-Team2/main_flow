import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print(f"경고: .env 파일을 찾을 수 없습니다. 경로: {dotenv_path}. 환경 변수가 직접 설정되었는지 확인하세요.")

class Config: 
     SECRET_KEY     = os.environ.get('SECRET_KEY') or '개발용-임시-시크릿-키-입니다-반드시-변경하세요'
     OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
     DEFAULT_MODEL  = os.environ.get('DEFAULT_MODEL')

     DLIB_LANDMARK_MODEL_PATH = None

     @staticmethod  
     def init_app(app): 
        project_root = os.path.dirname(os.path.abspath(__file__))
        app.config['DLIB_LANDMARK_MODEL_PATH'] = os.path.join(project_root, 'app', 'models', 'shape_predictor_68_face_landmarks.dat')
        
        app.config['OPENAI_API_KEY_SET'] = Config.OPENAI_API_KEY
        app.config['DEFAULT_MODEL']      = Config.DEFAULT_MODEL

        # --- API 키 설정 경고 메시지 수정 ---
        if not Config.OPENAI_API_KEY: # GEMINI_API_KEY 대신 OPENAI_API_KEY 확인
            if hasattr(app, 'logger'):
                app.logger.warning("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다. LLM 연동이 안 될 수 있습니다.")
            else:
                print("경고 (logger 없음): OPENAI_API_KEY가 설정되지 않았습니다.")
        
        if not os.path.exists(app.config['DLIB_LANDMARK_MODEL_PATH']):
            if hasattr(app, 'logger'):
                app.logger.error(f"Dlib 모델 파일을 찾을 수 없습니다: {app.config['DLIB_LANDMARK_MODEL_PATH']}")
            else:
                print(f"에러 (logger 없음): Dlib 모델 파일을 찾을 수 없습니다: {app.config['DLIB_LANDMARK_MODEL_PATH']}")