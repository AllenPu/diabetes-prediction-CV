import os
import subprocess
import pandas as pd


if __name__ == "__main__":
    label_set = [10, 11, 12, 14, 15, 20, 31, 41, 51, 60, 61, 62, 63, 64, 65, 70, 80, 88]
    df = pd.read_excel('file.xlsx')
    eye_list, label_list = [], []
    for index, row in df.iterrows():
        if 'nan' not in str(row['图片1']):
            eye_list.append(str(row['图片1']))
            label_list.append(row['图片1的ETDRS1'])
        if 'nan' not in str(row['图片2']):
            eye_list.append(str(row['图片2']))
            label_list.append(row['图片2的ETDRS2'])
    address, failed_address = [], []
    img_base = '/teams/dr_1765761962/Dataset/全部基线眼底镜图片2025.10(1).14上传'
    store_base = '/teams/dr_1765761962/program_data/raw_process/'
    j = 0
    for i in range(len(eye_list)):
        name = eye_list[i]
        result = subprocess.run(['find', img_base, '-name', name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        paths = result.stdout.split('\n')[0]
        if os.path.exists(paths):
            group = str(label_list[i])
            destination = store_base + group
            if not os.path.exists(paths):
                subprocess.run(['mkdir', destination])
            subprocess.run(['scp', '-r',  paths, destination])
            j += 1
        else:
            failed_address.append(paths) 
    print('*******************')
    print(f" Done processing ! Successfully move {j} items. Failed to move {len(failed_address)} items")
    print('*******************')
    print(failed_address)
    print('*******************')