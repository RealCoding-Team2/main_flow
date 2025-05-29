# phishing_detection_project/app/deepfake_detector/__init__.py
from flask import Blueprint

# 'deepfake_detector' 라는 이름으로 블루프린트 객체를 만듭니다.
# 이 이름은 app/__init__.py 에서 블루프린트를 등록할 때 사용됩니다.
# 여기서 bp 라는 이름으로 객체를 생성해야 app/__init__.py 에서 from .deepfake_detector import bp 로 가져올 수 있습니다.
bp = Blueprint('deepfake_detector', __name__) # <--- 이 부분이 중요합니다! bp 라는 이름으로 Blueprint 객체를 생성합니다.

# 이 블루프린트에 속한 라우트들을 정의한 routes.py 파일을 임포트합니다.
# 이렇게 해야 routes.py에 정의된 @bp.route(...) 들이 실제로 등록됩니다.
from . import routes