
     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                         IMPORTING PROJECT MODULES                          ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


import sys
sys.path.append(r"C:\Users\pm\OneDrive\Bureau\PROJET\Python projet\Turbulence_Analysis\Analysis_Program")


     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                              IMPORTING DATA                                ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


Data_path = r"C:\Users\pm\OneDrive\Bureau\PROJET\Documents Projet"  #      ---  Path to the data  ---

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#        Data Selection        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #        ---  Data from 6 September 2022 ---
'''
Data_name = "solo_L3_multi-mag-rpw-scm-merged-rtn-256_20220906_Vd0.cdf"
'''

    #        ---  Data from 4 April 2024 ---

Data_name = "solo_L3_multi-mag-rpw-scm-merged-rtn-256-cdag_20240404_Vd0.cdf"


    #        ---  Data from 9 August 2021 ---
'''
Data_name = "solo_L3_multi-mag-rpw-scm-merged-rtn-256-cdag_20210809_Vd0.cdf"
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#          Parameters          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

nperseg = 8192   #           ---  Number of points for each segment  ---

noverlap = nperseg*0.8   #      ---  Number of points to overlap between segments  ---

rotation = 'yes'   #      --- yes for field-aligned coordinates, no for RTN coordinates  ---


    # ────────────────────────    Données du 6 septembre 2022   ────────────────────────


if "20220906" in Data_name:

    low_filtred_limit = 70  #      ---  Lower limit for filtering (Figure 2, 4)  ---
    high_filtred_limit = 100  #     ---  Upper limit for filtering (Figure 2, 4)  ---


    low_limit_cyclo = 0.645  #      ---  Lower limit for filtering (Figure 4, 5)  ---
    high_limit_cyclo = 42  #     ---  Upper limit for filtering (Figure 4, 5)  ---
    low_limit_w = 7.75  #      ---  Lower limit for filtering (Figure 4, 5)  ---
    high_limit_w = 100    #     ---  Upper limit for filtering (Figure 4, 5)  ---


    start_time = "00:00:00"  #      ---  Starting time of the Data Selctioon  ---
    end_time = "23:59:00"  #       ---  Ending time of the Data Selctioon  ---


    # ────────────────────────    Données du 4 avril 2024   ────────────────────────


elif "20240404" in Data_name: 

    low_filtred_limit = 70
    high_filtred_limit = 100

    low_limit_cyclo = 0.65
    high_limit_cyclo = 4
    low_limit_w = 7.75
    high_limit_w = 100

    start_time = "00:00:00"
    end_time = "08:00:00" 


    # ────────────────────────    Données du 9 août 2021   ────────────────────────


elif "20210809"  in Data_name:

    low_filtred_limit = 1
    high_filtred_limit = 10
    
    low_limit_cyclo = 0.65
    high_limit_cyclo = 4
    low_limit_w = 4
    high_limit_w = 100

    start_time = "00:00:00"
    end_time = "08:00:00" 
