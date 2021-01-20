from prettytable import PrettyTable

def t(h,m,s):
    return 3600*h + 60*m + s

def a(t, h,m,s):
    return (3600*h + 60*m + s) - t

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
# 2046.88075 2753.651

# ECG-Phono-Seismo DAQ Data 8 20 2020 2
# D:  0
# 0.0 135.00925
# D:  10
# 477.7315 3275.023

laptop  = {"1 9 2020 AH TDMS ESSENTIAL" :{"file_time" : {0: (13, 27, 23),
                                                        10: (14,  0,  6),
                                                        20: (14,  6, 34),
                                                        30: (14, 11,  6),
                                                        40: (14, 16, 27)},
                                                        
                                          "data_time" : {0: 1.707,
                                                        10: 1096.6215,
                                                        20: 1471.61,
                                                        30: 1741.604,
                                                        40: 2046.88075},
                                                        
                                          "echo_time" : {0: (13, 44, 10),
                                                        10: (14,  3, 40),
                                                        20: (14, 10, 51),
                                                        30: (14, 15, 46),
                                                        40: (14, 29, 26)}},
                                                        
            "ECG-Phono-Seismo DAQ Data 8 20 2020 2" : {"file_time" : {   0: (13, 23,33),
                                                                        10: (13, 41, 41),
                                                                        20: (13, 41, 41),
                                                                        30: (13, 41, 41),
                                                                        40: (13, 41, 41)},
                                                        
                                                        "data_time" : {  0: 0.0,
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
    table.field_names = ["Time Difference on File Name", "Start Time of File Data"]
    for dosage in laptop[folder]["file_time"]:
        h, m, s = laptop[folder]["file_time"][dosage]

        if dosage == 0:
            sf = t(h, m, s)
            sd = laptop[folder]["data_time"][dosage]

            # print(len(sf), len(sd))
            table.add_row([0, sd])
        else:
            table.add_row([a(sf, h, m, s), laptop[folder]["data_time"][dosage] - sd])

        if dosage == 40:
            print(table)


