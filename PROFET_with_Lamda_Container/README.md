###### ECR 명령어 실행시 SUDO 추가로 꼭 붙여줘야함

#### 진행순서
##### 1. EC2에 mkdir Profet로 폴더만들기
##### 2. start.sh 그대로 세팅
##### 3. app.py , requirements.txt , feature_info.py , Dockerfile 만들기
##### 4. ECR 에서 리포지토리 생성
##### 5. 생성한 리포지토리 들어가서 푸시명령보기 클릭!
##### 6. 명령어에 SUDO 모두 붙여서 터미널에 입력하기 (맨처음 명령어는 DOCKER 앞에 SUDO 붙이기)
##### 7. api gateway 배포