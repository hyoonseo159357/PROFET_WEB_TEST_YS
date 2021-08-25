##### << 진행순서 >>
##### 1. EC2 생성후 EFS 마운트 (EFS에 람다에서 사용할 환경세팅 + 저장된 트레이닝된 모델을 불러 사용할것임)
##### 2. EFS 에 필요한 패키지들 설치
##### 3. EFS 에 모델 트레닝후 저장 ( EFS 에서 훈련시키지않고, 외부에서 훈련시켜서 가져오면, 파이썬 버전이나 패키지들의 버전이 맞지않아 람다에서 가져와 사용할수 없음)
##### 4. 람다에서 EFS환경+트레이닝된 모델을 사용하여 predict
##### 5. 버전을 만들고 프로비저닝하기
##### 6. api gateway로 연결하기