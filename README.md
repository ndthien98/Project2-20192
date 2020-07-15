# Tên model: Nhận diện thực thể có tên sử dụng mạng CNN + BiLSTM + CRFs
## Mục tiêu:
Nhận diện các thực thể có tên trong văn bản như tên người, tên địa danh, tên tổ chức.
## Công nghệ sử dụng
* Python 3.7
## Hướng dẫn cài đặt
Clone project:
>git clone https://github.com/ndthien98/Project2-20192

Mở foler src: 
> cd src

Cài đặt môi trường cần thiết.
> pip install -r requirements.txt

Đặt dữ liệu vào thư mục data:<br>
VLSP 2018: https://vlsp.org.vn/resources-vlsp2018 <br>
Global Vector: https://nlp.stanford.edu/projects/glove/
<br>
data/vi.train<br>
data/vi.cv<br>
data/vi.test<br>
data/glove.6B.100d.txt<br>


Chạy code thử nghiệm:
> python demo_original.py
