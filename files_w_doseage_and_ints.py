

'''
Conatins all patient's test folder names with associated dosage levels and viable intervals

The dictionary is set up as:
      Folder Name --> Dosage Level --> Intervals
'''
# Known files
files = { "1 9 2020 AH TDMS ESSENTIAL"   :   {0:  {1: {"file_name": "DAQData_010920132723",
                                                      "echo_time": 1016,
                                                      "intervals":   {
                                                                        1: [113, 133]
                                                                      }}},

                                             10: {1: {"file_name": "DAQData_010920140006",
                                                      "echo_time": 1318,
                                                      "intervals":   {
                                                                        1: [1226, 1254]
                                                                      }}},                                                                      
                                                                                                                                         
                                             20: {1: {"file_name": "DAQData_010920140634",
                                                      "echo_time": 1736,
                                                      "intervals":   {
                                                                         1: [1700, 1720]
                                                                        }}},                                        
                                             
                                             30: {
                                                  1: {"file_name": "DAQData_010920141106",
                                                      "echo_time": 2029,
                                                      "intervals": {
                                                                    1: [1995, 2015]
                                                                    }}},

                                             40: {
                                                  1: {"file_name": "DAQData_010920142816",
                                                      "echo_time": 2831,
                                                      "intervals": {
                                                                    1: [2740, 2750]
                                                                    }}}},
                                          
          "ECG-Phono-Seismo DAQ Data 8 20 2020 2"   :   {0: {
                                                             1 : {"file_name": "DAQData_082020133649",
                                                                  "echo_time": 334 ,
                                                                  "intervals": {
                                                                                1: [0, 15.5]
                                                                                }}},

                                                         10:  {
                                                               1: {"file_name": "DAQData_082020134141",
                                                                   "echo_time": 1037,
                                                                   "intervals": {   
                                                                                    1: [1126, 1154]
                                                                                }}},
                                                                                                  
                                                         20: {1: {"file_name": "DAQData_082020134141",
                                                                  "echo_time": 1388,
                                                                   "intervals": {   
                                                                                    1: [1473, 1493]
                                                                                    }}},


                                                         30: {1: {"file_name": "DAQData_082020134141",
                                                                  "echo_time": 1662,
                                                                   "intervals": {   
                                                                                    1: [1745, 1763]
                                                                                    }}},
                                                         
                                                         40: {1: {"file_name": "DAQData_082020134141",
                                                                  "echo_time": 2042 ,
                                                                   "intervals": {   
                                                                                    1: [2125, 2145]
                                                                   }}}}}
                                            