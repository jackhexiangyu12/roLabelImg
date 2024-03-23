import os
import xml.etree.ElementTree as ET

# 指定目录路径
directory = r'C:\Users\hxy\Downloads\14\14'

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        file_path = os.path.join(directory, filename)
        # 解析XML文件
        tree = ET.parse(file_path)
        root = tree.getroot()
        # 查找并修改person标签为people标签
        for obj in root.findall('.//object'):
            name = obj.find('name')
            if name is not None and name.text == 'person':
                name.text = 'people'
        # 保存修改后的XML文件
        tree.write(file_path)
