FROM almalinux:9.5
RUN dnf install -y python3.12 python3.12-pip
COPY requirements.txt output_schema.json /root
WORKDIR /root
RUN pip3.12 install -r requirements.txt
COPY forward_alerts.py forward_alerts.py
CMD ["python3.12", "forward_alerts.py"]