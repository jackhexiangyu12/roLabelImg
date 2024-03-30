import os
import random
import datetime
import time

def modify_xml_files(folder_path):
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    start_time = datetime.datetime(2024, 3, 20, 10, 33)
    end_time = datetime.datetime(2024, 3, 30, 17, 54)
    current_time = start_time
    while current_time < end_time:
        # Calculate a random time difference between 3 and 7 minutes
        time_diff = random.randint(3, 7)
        current_time += datetime.timedelta(minutes=time_diff)
        if random.random() < 1/30:
            # Introduce a 2-3 hour delay with 1/30 probability
            current_time += datetime.timedelta(hours=random.randint(2, 3))
        if current_time.hour >= 22 or current_time.hour < 6:
            # Skip night time
            continue
        # Set the modification time of the files
        for xml_file in xml_files:
            mod_time = time.mktime(current_time.timetuple())
            os.utime(os.path.join(folder_path, xml_file), (mod_time, mod_time))
        print('Modified ',,' at', current_time)

folder_path = r'C:\Users\hxy\Downloads\14\14_bak'
modify_xml_files(folder_path)
