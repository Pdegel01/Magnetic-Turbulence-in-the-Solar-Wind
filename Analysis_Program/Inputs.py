# ====================================== #
#       IMPORTING PROJECT MODULES        #
# ====================================== #

import sys
sys.path.append(r"C:\Users\pm\OneDrive\Bureau\PROJET\Python projet\Turbulence_Analysis\Analysis_Program")


# ====================================== #
#            IMPORTING DATA              #
# ====================================== #

nperseg = 8192*2
noverlap = nperseg*0.8
rotation = 'yes'
Data_path = r"C:\Users\pm\OneDrive\Bureau\PROJET\Documents Projet"

'''
# Data from 6 September 2022
Data_name = "solo_L3_multi-mag-rpw-scm-merged-rtn-256_20220906_Vd0.cdf"

'''

# Data from 4 April 2024
Data_name = "solo_L3_multi-mag-rpw-scm-merged-rtn-256-cdag_20240404_Vd0.cdf"

'''
# Data from 9 August 2021
Data_name = "solo_L3_multi-mag-rpw-scm-merged-rtn-256-cdag_20210809_Vd0.cdf"
'''
'''
start_time = "0:00:00"
end_time = "23:50:00" 
'''

start_time = "00:00:00"
end_time = "08:00:00" 


if "20220906" in Data_name:

    # Données du 6 septembre 2022
    low_limit_cyclo = 0.645
    high_limit_cyclo = 42
    low_limit_w = 7.75
    high_limit_w = 100

    low_filtred_limit = 70
    high_filtred_limit = 100
elif "20240404" in Data_name:

    # Données du 4 avril 2024
    low_limit_cyclo = 0.65
    high_limit_cyclo = 4
    low_limit_w = 7.75
    high_limit_w = 100

    low_filtred_limit = 70
    high_filtred_limit = 100
elif "20210809" in Data_name:

    # Données du 9 août 2021
    low_limit_cyclo = 0.65
    high_limit_cyclo = 4
    low_limit_w = 4
    high_limit_w = 100

    low_filtred_limit = 1
    high_filtred_limit = 10

# ====================================== #
#              PROCESSING                #
# ====================================== #