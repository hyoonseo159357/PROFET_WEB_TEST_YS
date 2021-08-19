# python3.8 lambda base image
FROM public.ecr.aws/lambda/python:3.8

# copy requirements.txt to container
COPY requirements.txt ./

# installing dependencies
RUN pip3 install --upgrade --ignore-installed pip setuptools
RUN pip3 install -r requirements.txt
RUN yum install git -y

# Copy function code to container
COPY feature_info.py ./
COPY app.py ./
RUN git clone https://github.com/hyoonseo159357/PROFET_Web_ys.git

# setting the CMD to your handler file_name.function_name
CMD [ "app.lambda_handler" ]