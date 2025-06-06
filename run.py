
from flask import render_template # 추가 임포트
from app import create_app        # 기존 create_app 함수 임포트
import os

application = create_app() # 기존 Flask 앱 인스턴스 생성

@application.route('/')
def index():
    # 템플릿 파일 경로 확인용 코드 추가
    template_dir  = os.path.join(application.root_path, application.template_folder)
    template_path = os.path.join(template_dir, 'index.html')
    
    # 로그 레벨을 INFO로 변경하여 항상 보이도록 함
    application.logger.info(f"[템플릿 경로 확인] Jinja2가 찾는 templates 폴더 경로: {template_dir}")
    application.logger.info(f"[템플릿 경로 확인] index.html 예상 절대 경로: {template_path}")
    application.logger.info(f"[템플릿 경로 확인] 실제로 index.html 파일 존재 여부: {os.path.exists(template_path)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    server_port   = int(os.environ.get('PORT', 5001))
    is_debug_mode = application.config.get('FLASK_DEBUG', True)
    
    application.logger.info(f"Flask 애플리케이션을 http://0.0.0.0:{server_port} 에서 시작합니다 (디버그 모드: {is_debug_mode}).")
    
    application.run(host='0.0.0.0', port=server_port, debug=is_debug_mode)