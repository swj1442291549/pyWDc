import pandas as pd

from wd import LC, Model

if __name__ == "__main__":
    lc_data = pd.read_csv("test/test_lc.csv")
    output_folder = "testresult"
    lc = LC(lc_data, IBAND=7, WLA=0.551, AEXTINC=0.14, CALIB=0.36895)
    mod = Model([lc], "test", PERIOD=0.372382, TAVH=0.571711)
    mod.prepare_run()
    mod.clean_lc_gp()
    mod.cal_res_curve_test()
    try:
        mod.find_q_best()
        mod.cal_bol_mag_abs(dm=9.476)
        mod.correct_temp_comb()
        mod.cal_bol_mag_abs(dm=9.476)
    except:
        pass
    else:
        mod.cal_fit()
        mod.save(output_folder)
    mod.clean_run()
