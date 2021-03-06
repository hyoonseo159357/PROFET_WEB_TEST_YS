sudo yum update
aws configure

# docker-compose 설치
sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# docker-compose 실행 권한 부여
sudo chmod +x /usr/local/bin/docker-compose

# docker 설치
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
