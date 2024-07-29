import gdown

file_id = '1jh2p2DlaWsDo_vEWIcTrNh3mUuXd-cw6'

url = f'https://drive.google.com/uc?id={file_id}'

output = 'vi_text_retrieval.csv'

gdown.download(url, output, quiet=False)
