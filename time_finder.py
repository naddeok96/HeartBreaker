from prettytable import PrettyTable

def get_seconds(time):
    h,m,s = time
    return 3600*h + 60*m + s

def get_time(seconds):
    h = int(seconds/3600)
    m = int((seconds- h*3600)/60)
    s = seconds- h*3600 - m*60
    return (h,m,s)

def get_diff(file_time, echo_time, data_time):

    file_sec = get_seconds(file_time)
    echo_sec = get_seconds(echo_time)

    diff_sec = (echo_sec + 7) - file_sec
    diff_time = get_time(diff_sec)

    time_to_use = data_time + diff_sec
    return diff_time, time_to_use


# 1 9 2020 AH TDMS ESSENTIAL
# D:  0
# 1.707 1096.62125
# D:  10
# 1096.6215 1471.60975
# D:  20
# 1471.61 1741.60375
# D:  30
# 1741.604 2046.8805
# D:  40
# 2753.65125 3889.8105

# ECG-Phono-Seismo DAQ Data 8 20 2020 2
# D:  0
# 189.3175 477.73125
# D:  10
# 477.7315 3275.023

laptop  = {"1 9 2020 AH TDMS ESSENTIAL" :{"file_time" : {0: (13, 27, 23),
                                                        10: (14,  0,  6),
                                                        20: (14,  6, 34),
                                                        30: (14, 11,  6),
                                                        40: (14, 28, 16)},
                                                        
                                          "data_time" : {0: 1.707,
                                                        10: 1096.6215,
                                                        20: 1471.61,
                                                        30: 1741.604,
                                                        40: 2753.65125},
                                                        
                                          "echo_time" : {0: (13, 44, 10),
                                                        10: (14,  3, 40),
                                                        20: (14, 10, 51),
                                                        30: (14, 15, 46),
                                                        40: (14, 29, 26)}},
                                                        
            "ECG-Phono-Seismo DAQ Data 8 20 2020 2" : {"file_time" : {   0: (13, 36, 49),
                                                                        10: (13, 41, 41),
                                                                        20: (13, 41, 41),
                                                                        30: (13, 41, 41),
                                                                        40: (13, 41, 41)},
                                                        
                                                        "data_time" : {  0: 189.3175,
                                                                        10: 477.7315,
                                                                        20: 477.7315,
                                                                        30: 477.7315,
                                                                        40: 477.7315},
                                                                        
                                                        "echo_time" : {  0: (13, 32, 10),
                                                                        10: (13, 50, 53),
                                                                        20: (13, 56, 44),
                                                                        30: (14,  1, 18),
                                                                        40: (14,  7, 38)}}}

for folder in laptop:
    print(folder)
    table = PrettyTable()
    table.field_names = ["Time on File", "Time of 2D Echo", "Time Difference", "Time to Use"]
    for dosage in laptop[folder]["file_time"]:
        file_time = laptop[folder]["file_time"][dosage]
        data_time = laptop[folder]["data_time"][dosage]
        echo_time = laptop[folder]["echo_time"][dosage]
        
        
        sec = get_seconds(file_time)
        time = get_time(sec)
        
        diff_time, time_to_use = get_diff(file_time, echo_time, data_time)
        
        table.add_row([file_time, echo_time, diff_time, time_to_use])
        
        if dosage == 40:
            print(table)