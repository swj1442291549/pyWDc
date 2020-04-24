import pandas as pd

from wd import LC, Model, RV

if __name__ == "__main__":
    lc_data = pd.read_csv("test/test_lc.csv")
    rv_data = pd.read_csv("test/test_rv1.csv")
    output_folder = "testresult_rv"
    lc = LC(lc_data, IBAND=7, WLA=0.551, AEXTINC=0.14, CALIB=0.36895)

    rv = RV(rv_data, 1)
    mod = Model([lc], [rv], "test", PERIOD=0.372382, TAVH=0.571711)
    mod.prepare_run()
    mod.clean_lc_gp()
    mod.cal_qout_cuvre()
    mod.plot_qout()
    mod.find_q_best()
    mod.cal_bol_mag_abs(dm=9.476)
    mod.correct_temp_comb()
    mod.cal_bol_mag_abs(dm=9.476)
    mod.cal_fit()
    mod.save(output_folder)
    mod.clean_run()
