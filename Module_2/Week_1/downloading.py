import gdown

file_id1 = '1i9dqan21DjQoG5Q_VEvm0LrVwAlXD0vB'
file_id2 = '1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq'

url1 = f'https://drive.google.com/uc?id={file_id1}'
url2 = f'https://drive.google.com/uc?id={file_id2}'

output1 = 'dog.jpeg'
output2 = 'advertising.csv'

gdown.download(url1, output1, quiet=False)
gdown.download(url2, output2, quiet=False)