# YOLO 모델 학습 및 SEJONG.PY 실행을 위한 요구 사항

## 아나콘다 설치 안내
# 특정 버전의 Python을 사용하여 가상 환경 생성
conda create -n '원하는 가상환경 이름' python==3.10.13

## 필수 패키지 설치

# 생성한 가상 환경 활성화
conda activate '원하는 가상환경 이름'

# YOLO 모델 및 SEJONG.PY 실행에 필요한 패키지 설치
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install torch==1.11.0
pip install torchvision==0.12.0
pip install numpy==1.23.5

## 모델 가중치 및 실행 방법
# 'best_model.pth' 및 'best.pt' 파일을 다운로드한 후, 코드 내에서 정확한 경로를 지정해야 합니다.

# 코드 130번 줄:
# 모델 가중치 로드 (학습된 모델 가중치 경로를 지정해야 함)
# model.load_state_dict(torch.load('best_model.pth 파일의 경로', map_location=device))

# 코드 201번 줄:
# YOLO 모델 로드 및 객체 탐지 실행
# model_yolo = YOLO('best.pt 파일의 경로')

## SEJONG.PY 실행 명령어
# 데이터 디렉토리 경로와 결과 디렉토리를 지정하여 SEJONG.PY 실행
python SEJONG.py --data_dir '데이터 디렉토리 경로' --result_dir ./result
# 주의: 결과 디렉토리가 없으면 생성됩니다. 결과를 저장할 다른 경로도 지정할 수 있습니다.
***  중요사항 
SEJONG.py 파일과 model 폴더가 같은 디렉토리나 위치에 있으면 명령어 돌아갑니다.
