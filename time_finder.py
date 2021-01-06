
times = {"Hr/Min/Sec": {"0"  :   (13, 32, 10),
                        "10" :   (13, 50, 53), 
                        "20" :   (13, 56, 44),
                        "30" :   (14,  1, 18),
                        "40" :   (14,  7, 38),
                        "40A":   (14, 11, 41), 
                        "Post" : (14, 25, 46)},

        "Computer_Time" : {},

        "File_Times" : {"1" : (0        , 135.00925),
                        "2" : (135.0095 , 184.51550),
                        "3" : (184.51575, 189.31725),
                        "4" : (189.31725, 477.73125),
                        "5" : (477.7315 , 3275.0230)}}

def hr_min_sec_2_scalar(times):

    lead_time_of_laptop = 7 # [s]

    # Cycle through all doses
    for dose in times["Hr/Min/Sec"]:
        # Load time in Hr/Min/Sec
        time_in_hr_min_sec = times["Hr/Min/Sec"][dose]

        # Convert to Seconds
        if dose == "0":
            times["Computer_Time"][dose] = 0
            start_time = 7 + (60 * (60 * time_in_hr_min_sec[0] + time_in_hr_min_sec[1]) + time_in_hr_min_sec[2])
            
        else:
            time_in_sec = 60 * (60 * time_in_hr_min_sec[0] + time_in_hr_min_sec[1]) + time_in_hr_min_sec[2]
            times["Computer_Time"][dose] = time_in_sec - start_time

        


hr_min_sec_2_scalar(times)
print(times["Computer_Time"])