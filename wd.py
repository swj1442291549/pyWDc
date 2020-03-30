import os
import re
import pickle
import signal
import subprocess

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.interpolate import interp1d
from astroML.time_series import MultiTermFit

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def init_run():
    """Build the source code

    The source code has changed to fit A without velocity
    """
    print("Compiling ...")
    subprocess.call(
        "cd material/lcdc; gfortran -o lc lc14june2015flop.f; gfortran -o dc dc22may2015flop.f",
        shell=True,
    )


class LC:
    def __init__(self, data, IBAND, WLA, AEXTINC, CALIB):
        """Light Curve

        The data must has following columns:
            phase, mag, (jd), (mag_err)

        Args:
            data (DataFrame): Light curve data
            IBAND (int): Identifier in Table 2
            WLA (float): Observational wavelengths in microns
            AEXTINC (float): Interstellar extinction A in magnitude
            CALIB (float): Flux calibration constant in Table 3

        """
        self.data = data
        self.IBAND = IBAND
        self.WLA = WLA
        self.AEXTINC = AEXTINC
        self.CALIB = CALIB
        self.check_err()

    def cal_phase(self, HJD0, PERIOD):
        """Calcualte the phase from JD

        Args:
            HJD0 (float): Ephemeris reference time
            PERIOD (float): period
        """
        phase = np.fmod(self.data.jd - HJD0, PERIOD) / PERIOD
        self.data = self.data.assign(phase=phase)

    def sort_phase(self):
        """Sort the order based on phase"""
        self.data.sort_values("phase", inplace=True)
        self.data.reset_index(inplace=True, drop=True)

    def check_err(self):
        if "mag_err" not in self.data.columns:
            self.data = self.data.assign(mag_err=1)

class RV:
    def __init__(self, data, mntype):
        """Init

        Args:
            data (DataFrame): rv data frame
            mntype (int): Star ID, 1 if same as input temperature in Model
        """
        self.data = data
        self.mntype = mntype
        self.check_err()

    def cal_phase(self, HJD0, PERIOD):
        """Calcualte the phase from JD

        Args:
            HJD0 (float): Ephemeris reference time
            PERIOD (float): period
        """
        phase = np.fmod(self.data.jd - HJD0, PERIOD) / PERIOD
        self.data = self.data.assign(phase=phase)

    def sort_phase(self):
        """Sort the order based on phase"""
        self.data.sort_values("phase", inplace=True)
        self.data.reset_index(inplace=True, drop=True)

    def check_err(self):
        if "v_err" not in self.data.columns:
            self.data = self.data.assign(v_err=1)


class Model:
    def __init__(self, lc_list, directory, PERIOD, TAVH, rv_list=None):
        """Model

        Args:
            directory (string): output directory
            lc_list (list): list of light curves
            PERIOD (float): period
            TAVH (float): Effective temperature in 10000K
        """
        self.lc = lc_list
        self.rv = rv_list
        self.NLC = len(lc_list)
        self.IFVC1 = 0
        self.IFVC2 = 0
        if self.rv != None:
            for rv in self.rv:
                if rv.mntype == 1:
                    self.IFVC1 = 1
                else:
                    self.IFVC2 = 1
        self.PERIOD = PERIOD
        self.temp_color = TAVH
        self.TAVH = TAVH
        self.TAVC = TAVH
        self.VUNIT = 100
        self.cal_hjd0()
        self.cal_phase()
        self.cal_pshift()
        self.directory = directory

    def cal_hjd0(self):
        """Calculate HJD0

        If LC only has phase, pick an arbitary HJD0. Else, pick the minimum of JD.
        """
        has_jd = False
        hjd0_lc = None
        hjd0_rv = None
        if self.NLC > 0:
            if "jd" in self.lc[0].data.columns:
                has_jd = True
                hjd0_lc = min([min(lc.data.jd) for lc in self.lc])
        if self.IFVC1 + self.IFVC2 > 0:
            if "jd" in self.rv[0].data.columns:
                has_jd = True
                hjd0_rv = min([min(rv.data.jd) for rv in self.rv])
        if not has_jd:
            self.HJD0 = 50000.0
        else:
            self.HJD0 = np.min([hjd0_lc, hjd0_rv])

    def cal_phase(self):
        """Calculate PHASE"""
        if self.NLC > 0:
            for lc in self.lc:
                if "phase" in lc.data.columns:
                    lc.sort_phase()
                else:
                    lc.cal_phase(self.HJD0, self.PERIOD)
                    lc.sort_phase()
        if self.IFVC1 + self.IFVC2 > 0:
            for rv in self.rv:
                if "phase" in rv.data.columns:
                    rv.sort_phase()
                else:
                    rv.cal_phase(self.HJD0, self.PERIOD)
                    rv.sort_phase()

    def cal_pshift(self):
        """Calculate PSHIFT

        Use the average of nearby 5 data points to determine the largest peak
        """
        pshift_0 = self.lc[0].data.phase.iloc[
            self.lc[0].data.mag.shift(periods=1).rolling(5).mean().idxmax()
        ]
        pshift_list = list()
        for lc in self.lc:
            pshift = lc.data.phase.iloc[
                lc.data.mag.shift(periods=1).rolling(5).mean().idxmax()
            ]
            if np.abs(pshift - pshift_0) > 0.7:
                pshift += -1 if pshift > pshift_0 else 1
            pshift_list.append(pshift)
        self.PSHIFT = np.mean(pshift_list)

    def cal_potential(self, q):
        """Calculate the potential for Star 1 and Star 2

        Interpolate the results from DOI:10.1186/s40668-015-0008-8

        Args:
            q (float): mass-ratio
        """
        df = pd.read_csv(
            "material/RocheTable.txt",
            delim_whitespace=True,
            names=[
                "rm",
                "rl1",
                "omega1",
                "x2",
                "omega2",
                "x3",
                "omega3",
                "rbk",
                "ry",
                "rz",
                "area",
                "vol",
                "req",
            ],
        )
        interp1 = interp1d(df.rm, df.omega1)
        interp2 = interp1d(df.rm, df.omega2)
        p1 = float(interp1(q))
        p2 = float(interp2(q))
        return p1, p2

    def plot_lc(self, save_path=None):
        """Plot the light curve"""
        if self.IFVC1 + self.IFVC2 == 0:
            fig, ax = plt.subplots()
        else:
            fig, (ax, ax_rv) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )
        for lc in self.lc:
            if "flag" not in lc.data.columns:
                phase = np.concatenate(
                    [lc.data.phase - 1, lc.data.phase, lc.data.phase + 1]
                ) # - self.PSHIFT
                mag = np.concatenate([lc.data.mag, lc.data.mag, lc.data.mag])
                ax.scatter(phase, mag, s=4, label="{0}".format(lc.IBAND))
            else:
                data_sel = lc.data[lc.data.flag == True]
                phase = np.concatenate(
                    [data_sel.phase - 1, data_sel.phase, data_sel.phase + 1]
                ) # - self.PSHIFT
                mag = np.concatenate([data_sel.mag, data_sel.mag, data_sel.mag])
                ax.scatter(phase, mag, s=4, label="{0}".format(lc.IBAND))
                data_sel = lc.data[lc.data.flag == False]
                phase = np.concatenate(
                    [data_sel.phase - 1, data_sel.phase, data_sel.phase + 1]
                ) # - self.PSHIFT
                mag = np.concatenate([data_sel.mag, data_sel.mag, data_sel.mag])
                ax.scatter(phase, mag, s=8, label="__nolegend__", marker="x")
        if hasattr(self, "fit"):
            for df in self.fit:
                phase = np.concatenate(
                    [df.phase - 1, df.phase, df.phase + 1]
                ) # - self.PSHIFT
                mag = np.concatenate([df.magd, df.magd, df.magd])
                ax.plot(phase, mag, label="__nolegend__", c="red")
        ax.set_xlabel(r"Phase")
        ax.set_ylabel(r"Magnitude (mag)")
        ax.set_xlim(-0.2, 1.2)
        ax.invert_yaxis()
        ax.legend()
        rv_color_list = ['red', 'blue']
        if self.IFVC1 + self.IFVC2 != 0:
            for rv in self.rv:
                phase = np.concatenate(
                    [rv.data.phase - 1, rv.data.phase, rv.data.phase + 1]
                ) # - self.PSHIFT
                v = np.concatenate([rv.data.v, rv.data.v, rv.data.v])
                ax_rv.scatter(phase, v, label="{0:d}".format(rv.mntype), s=4, c=rv_color_list[rv.mntype - 1])
            if hasattr(self, "fit_rv"):
                phase = np.concatenate(
                    [self.fit_rv.phase - 1, self.fit_rv.phase, self.fit_rv.phase + 1]
                ) # - self.PSHIFT
                v1 = np.concatenate([self.fit_rv.v1, self.fit_rv.v1, self.fit_rv.v1])
                v2 = np.concatenate([self.fit_rv.v2, self.fit_rv.v2, self.fit_rv.v2])
                ax_rv.plot(phase, v1, label='__nolegend__', c=rv_color_list[0], alpha=0.5)
                ax_rv.plot(phase, v2, label='__nolegend__', c=rv_color_list[1], alpha=0.5)
            ax_rv.set_ylabel(r"$v_r$ (km/s)")
            ax_rv.set_xlabel(r"Phase")
            ax_rv.legend()
            plt.subplots_adjust(hspace=0)
        if save_path:
            p = Path("fig/{0}".format(save_path))
            if not p.is_dir():
                p.mkdir(parents=True)
            plt.savefig("fig/{0}/{1}_lc.pdf".format(save_path, self.directory))
            plt.close()
        else:
            plt.show()

    def prepare_run(self):
        """Prepare for the run"""
        run_dir = Path("run/{0}".format(self.directory))
        if not run_dir.is_dir():
            run_dir.mkdir(parents=True)
        subprocess.call("cp material/lcdc/* {0}".format(run_dir), shell=True)

    def generate_dcin(self, RM, A=2, XINCL=80, NITERS=10, XLAMDA=1e-5, is_rm_fix=True):
        """Generate dcin.active
        
        Args:
            RM (flaot): mass ratio
            A (float): separation
                       (2)
            XINCL (float): inclination angle
                           (80)
            NITERS (int): number of iteration
                          (10)
            XLAMDA (float): precision
                            (1e-5)
            is_rm_fix (Boolen): whether RM is fix
                               (True)
        """
        with open("run/{0}/dcin.active".format(self.directory), "w") as f:
            f.write(
                " +0.2d-1 +0.2d-1 +1.0d-3 +0.2d-1 +0.2d-1 +1.0d-3 +0.1d-3 +0.2d-1\n"
            )
            f.write(
                " +5.0d-2 +1.0d-3 +1.0d-2 +1.0d-2 +1.0d-2 +2.0d-3 +2.0d-1 +1.0d-2 +1.0d-3 +2.0d-2 +2.0d-2\n"
            )
            f.write(
                " +5.0d-2 +5.0d-2 +2.0d-2 +2.0d-2 +0.3d-2 +1.0d-2 +1.0d-2 +1.0d-2 +1.0d-2\n"
            )
            f.write(
                " 1111 1111 1111101 01110 1101{0} 11111 11111 11111 11111 11111 11111 01111 {1:0>2d}{2:10.3e}{3:6.3f}\n".format(
                    int(is_rm_fix), NITERS, XLAMDA, 1.0
                )
            )  # q, NITERS, XLAMDA, VLR
            f.write("  1  0  2  0\n")
            f.write("{0:d} {1:d} {2:0>2d} 0 2 0 0 1 1 1 0\n".format(self.IFVC1, self.IFVC2, self.NLC))  # NLC
            f.write("1 1 1 1 1 1 0 -2 -2 0 1 1 0 1 0  0.0000\n")
            f.write(
                "{0}{1:15.6f}{2:17.10e}  0.000000D+00{3:10.4f} 0.00000  1\n".format(
                    2, self.HJD0, self.PERIOD, self.PSHIFT
                )
            )  # HJD0, PZERO, PSHIFT
            f.write(
                    " 3 0 1 1  30  30  10  10{0:13.6f}{1:13.5e} 0.00000{2:9.3f}\n".format(
                    0, 0, self.VUNIT
                )
            )  # PERR0, DPERDT
            f.write(
                ".00000{0:13.6e}   1.0000    1.0000    0.0000{1:9.3f}{2:7.3f}{3:7.3f}   0.00    1.0000    1.0000\n".format(
                    A if not hasattr(self, "A") else self.A,
                    XINCL if not hasattr(self, "XINCL") else self.XINCL,
                    0.32 if self.TAVH < 0.62 else 1,
                    0.32 if self.TAVC < 0.62 else 1,
                )
            )  # A, XINCL, GR
            f.write(
                "{0:7.4f}{1:8.4f}{2:7.3f}{3:7.3f}{4:13.6e}{5:13.6e}{6:13.6e}  0.647  0.644  0.179  0.163{7:9.5f}\n".format(
                    self.TAVH,
                    self.TAVC,
                    0.5 if self.TAVH < 0.62 else 1,
                    0.5 if self.TAVC < 0.62 else 1,
                    self.PHSV,
                    self.PCSV,
                    RM,
                    1,
                )
            )  # TAVH, PHSV, PCSV, RM, DPCLOG
            f.write(
                "0.000000D+00 0.0000000D+02    0.00000 0.000000 0.0000000       0.00000000\n"
            )
            for i in range(self.IFVC1 + self.IFVC2):
                f.write(
                    "  7 1.000000D+00 2.000000D+00 -0.029  0.669  0.726  0.285 0.000D+00 0.00000D+00 0.12000 0.44000 0.55000 0.95000  0.440000 1\n")
            for lc in self.lc:
                f.write(
                    "{0:3d} 1.000000D+00 2.000000D+00 -0.029  0.669  0.726  0.285  0.0000D+00 0.000D+00 0 0.00000D+00 0.12000 0.44000 0.55000 0.95000 1\n".format(
                        lc.IBAND
                    )
                )  # IBAND
            for lc in self.lc:
                f.write(
                    "{0:9.6f}{1:8.4f} 1.0000D+00{2:12.5e}\n".format(
                        lc.WLA, lc.AEXTINC, lc.CALIB
                    )
                )  # WLA, AEXTINC, CALIB
            f.write("300.00000\n")
            f.write("300.00000\n")
            f.write("150.\n")
            if self.IFVC1 + self.IFVC2 > 0:
                for rv in self.rv:
                    for i in range(len(rv.data)):
                        item = rv.data.iloc[i]
                        f.write(
                            "{0:14.5f}{1:11.6f}{2:8.3f}\n".format(
                                item.phase, item.v / self.VUNIT, 1 / (item.v_err) ** 2
                            )
                        )
                    f.write("{0:14.5f}{1:11.6f}{2:8.3f}\n".format(-10001, 0, 0))
            for j, lc in enumerate(self.lc):
                if "flag" not in lc.data.columns:
                    for i in range(len(lc.data)):
                        item = lc.data.iloc[i]
                        f.write(
                            "{0:14.5f}{1:11.6f}{2:8.3f}\n".format(
                                item.phase, item.mag, 1 / item.mag_err ** 2
                            )
                        )
                else:
                    data_sel = lc.data[lc.data.flag == True]
                    for i in range(len(data_sel)):
                        item = data_sel.iloc[i]
                        f.write(
                            "{0:14.5f}{1:11.6f}{2:8.3f}\n".format(
                                item.phase, item.mag, 1 / item.mag_err ** 2
                            )
                        )
                if j != len(self.lc) - 1:
                    f.write("{0:14.5f}{1:11.6f}{2:8.3f}\n".format(-10001, 0, 0))
            f.write("  -10001.\n")
            f.write(" 2\n")

    def run_dc(self, RM, A=2, alarm_time=0, is_rm_fix=True):
        """Run DC command

        Save PID for method kill_dc
        
        Args:
            RM (float): mass ratio
        """
        signal.signal(signal.SIGALRM, timeout_handler)
        if alarm_time == 0:
            if is_rm_fix:
                signal.alarm(40 * self.NLC)
            else:
                signal.alarm(100 * self.NLC)
        else:
            signal.alarm(alarm_time * self.NLC)
        self.PHSV, self.PCSV = self.cal_potential(RM)
        self.clean_active()
        self.generate_dcin(RM=RM, is_rm_fix=is_rm_fix, A=A)
        try:
            pro = subprocess.Popen(
                "cd run/{0};./dc".format(self.directory),
                shell=True,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self.pid = pro.pid
            pro.communicate()
        except TimeoutException:
            self.kill_dc()
        else:
            signal.alarm(0)

        p = Path("run/{0}/lcin.input_from_DC".format(self.directory))
        return p.is_file()

    def kill_dc(self):
        """Kill DC"""
        os.killpg(os.getpgid(self.pid), signal.SIGTERM)

    def read_qout_lcin(self):
        with open("run/{0}/lcin.input_from_DC".format(self.directory), "r") as f:
            lines = f.readlines()
            return self.tf_number(lines[5].split()[6])

    def read_dcout(self):
        """Read mean residual predicted from dcout

        Returns:
            res (float): residual
        """
        with open("run/{0}/dcout.active".format(self.directory), "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            if "determinant" in lines[-i - 1]:
                break
        split = [item for item in re.split("\s|\n", lines[-i]) if item != ""]
        try:
            res = self.tf_number(split[1])
        except:
            pass
        else:
            return res

    def cal_res(self, q, is_q_fix=True, A=2, alarm_time=0):
        """Calculate the residual useing DC

        Args:
            q (float): mass-ratio
            is_q_fix (Boolen): fix RM or not
        """
        signal.signal(signal.SIGALRM, timeout_handler)
        if alarm_time == 0:
            if is_q_fix:
                signal.alarm(40 * self.NLC)
            else:
                signal.alarm(100 * self.NLC)
        else:
            signal.alarm(alarm_time * self.NLC)
        self.PHSV, self.PCSV = self.cal_potential(q)
        self.generate_dcin(RM=q, is_rm_fix=is_q_fix, A=A)
        try:
            self.run_dc()
        except TimeoutException:
            self.kill_dc()
            return
        else:
            signal.alarm(0)
            return self.read_dcout()

    def cal_res_curve(self):
        """Calculate the residual curve"""
        q_array = np.append(
            np.append(np.arange(0.05, 0.5, 0.02), np.arange(0.5, 2, 0.05)),
            np.arange(2, 10, 0.25),
        )
        if hasattr(self, "q"):
            q_list = list(self.res.q)
            res_list = list(self.res.res)
        else:
            q_list = list()
            res_list = list()
        alarm_time = 30 * self.NLC
        while len(res_list) < 50 and alarm_time <= 120 * self.NLC:
            for q in q_array:
                if q not in q_list:
                    if self.run_dc(q, alarm_time=alarm_time):
                        res = self.read_dcout()
                        if not isinstance(res, type(None)):
                            if not np.isnan(res):
                                q_list.append(q)
                                res_list.append(res)
            alarm_time *= 2
        self.res = pd.DataFrame({"q": q_list, "res": res_list}).sort_values("q")

    def cal_qout_curve(self):
        """Calculate the qout curve"""
        q_array = np.append(
            np.append(np.arange(0.05, 0.5, 0.02), np.arange(0.5, 2, 0.05)),
            np.arange(2, 10, 0.25),
        )
        q_list = list()
        qout_list = list()
        alarm_time = 30 * self.NLC
        while len(qout_list) < 50 and alarm_time <= 120 * self.NLC:
            for q in q_array:
                print(q)
                if q not in q_list:
                    if self.run_dc(q, alarm_time=alarm_time, is_rm_fix=False):
                        qout = self.read_lcin()
                        if not isinstance(qout, type(None)):
                            if not np.isnan(qout):
                                q_list.append(q)
                                qout_list.append(qout)
            alarm_time *= 2
        self.qout = pd.DataFrame({"q": q_list, "qout": qout_list}).sort_values("q")

    def cal_res_curve_test(self):
        """Calculate the residual curve for test"""
        q_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 3, 5])
        if hasattr(self, "q"):
            q_list = list(self.res.q)
            res_list = list(self.res.res)
        else:
            q_list = list()
            res_list = list()
        alarm_time = 30 * self.NLC
        while len(res_list) < 5 and alarm_time <= 120 * self.NLC:
            for q in q_array:
                print("q: {0:.1f}".format(q))
                if q not in q_list:
                    if self.run_dc(q, alarm_time=alarm_time):
                        res = self.read_dcout()
                        if not isinstance(res, type(None)):
                            if not np.isnan(res):
                                q_list.append(q)
                                res_list.append(res)
            alarm_time *= 2
        self.res = pd.DataFrame({"q": q_list, "res": res_list}).sort_values("q")

    def cal_error(self, q):
        if self.run_dc(q, is_rm_fix=False, alarm_time=500):
            with open("run/{0}/dcout.active".format(self.directory), "r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if "Input-Output in F Format" in lines[-i - 1]:
                    break
            try:
                split = [
                    item for item in re.split("\s|\n", lines[-i + 2]) if item != ""
                ]
                self.PSHIFT_ERR = self.tf_number(split[-1])
                split = [
                    item for item in re.split("\s|\n", lines[-i + 3]) if item != ""
                ]
                self.XINCL_ERR = self.tf_number(split[-1])
                split = [
                    item for item in re.split("\s|\n", lines[-i + 4]) if item != ""
                ]
                self.TAVC_ERR = self.tf_number(split[-1])
                split = [
                    item for item in re.split("\s|\n", lines[-i + 5]) if item != ""
                ]
                self.P_ERR = self.tf_number(split[-1])
                split = [
                    item for item in re.split("\s|\n", lines[-i + 6]) if item != ""
                ]
                self.RM_ERR = self.tf_number(split[-1])
                split = [
                    item for item in re.split("\s|\n", lines[-i + 7]) if item != ""
                ]
                self.l_ratio_ERR = self.tf_number(split[-1])
                self.is_final = True
            except:
                self.is_final = False

    def plot_res(self, save_path=None):
        fig, ax = plt.subplots()
        ax.plot(self.res.q, self.res.res)
        ax.scatter(self.res.q, self.res.res, s=4)
        if hasattr(self, "q_best"):
            res_sel = self.res[self.res.q == self.q_best]
            ax.scatter(res_sel.q, res_sel.res, c="r", s=6, zorder=100)
        ax.set_xlabel(r"q")
        ax.set_ylabel(r"Residual")
        if save_path:
            p = Path("fig/{0}".format(save_path))
            if not p.is_dir():
                p.mkdir(parent=True)
            plt.savefig("fig/{0}/{1}_res.pdf".format(save_path, self.directory))
            plt.close()
        else:
            plt.show()

    def plot_qout(self, save_path=None):
        fig, ax = plt.subplots()
        ax.plot(self.qout.q, self.qout.qout)
        ax.scatter(self.qout.q, self.qout.qout, s=4)
        if hasattr(self, "q_best"):
            ax.hlines(self.q_best, min(self.qout.q), max(self.qout.q))
        ax.set_xlabel(r"q")
        ax.set_ylabel(r"qout")
        if save_path:
            p = Path("fig/{0}".format(save_path))
            if not p.is_dir():
                p.mkdir(parent=True)
            plt.savefig("fig/{0}/{1}_qout.pdf".format(save_path, self.directory))
            plt.close()
        else:
            plt.show()

    def find_q_best(self):
        if hasattr(self, "qout"):
            self.q_best = np.median(self.qout.qout)
        else:
            res_sel = self.res[self.res.res < np.percentile(self.res.res, 25)]
            res_sort = self.res.sort_values("res")
            for position in range(len(self.res)):
                index = res_sort.iloc[position].name
                if index == 0 or index == len(self.res) - 1:
                    continue
                elif (
                    self.res.loc[index - 1].q not in res_sel.q.values
                    and self.res.loc[index + 1].q not in res_sel.q.values
                ):
                    continue
                else:
                    break
            self.q_best = self.res.loc[index].q

    def save(self, prefix=""):
        """Save the result to pkl"""
        result_dir = "result/{0}".format(prefix)
        p = Path(result_dir)
        if not p.is_dir():
            p.mkdir(parents=True)
        pickle.dump(self, open("{0}/{1}.pkl".format(result_dir, self.directory), "wb"))

    @staticmethod
    def load(filename, prefix=""):
        result_dir = "result/{0}".format(prefix)
        mod = pickle.load(open("{0}/{1}.pkl".format(result_dir, filename), "rb"))
        return mod

    def run_lc(self):
        """Run LC command"""
        with open("run/{0}/lcin.input_from_DC".format(self.directory), "r") as f:
            lines = f.readlines()
        with open("run/{0}/lcin.active".format(self.directory), "w") as f:
            for i, line in enumerate(lines):
                if "-0.200000    1.000000" not in line:
                    f.write(line)
                else:
                    f.write(line.replace("-0.200000", " 0.000000"))
        subprocess.call(
            "cd run/{0};./lc".format(self.directory),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read_lcout(self):
        with open("run/{0}/lcout.active".format(self.directory), "r") as f:
            lines = f.readlines()
        start_line_list = list()
        mpage_list = list()
        for i, line in enumerate(lines):
            if "mpage" in line and int(lines[i + 1].split()[0]) < 3:
                start_line_list.append(i)
                mpage_list.append(int(lines[i + 1].split()[0]))

        split = [
            item for item in re.split("\s|\n", lines[4 + start_line_list[0]]) if item != ""
        ]
        self.PSHIFT = self.tf_number(split[4])
        split = [
            item for item in re.split("\s|\n", lines[13 + start_line_list[0]]) if item != ""
        ]
        XINCL = self.tf_number(split[5])
        self.XINCL = XINCL if XINCL <= 90 else 180 - XINCL
        split = [
            item for item in re.split("\s|\n", lines[16 + start_line_list[0]]) if item != ""
        ]
        self.TAVC = self.tf_number(split[1])
        self.P = self.tf_number(split[4])
        self.q = self.tf_number(split[6])

        for i, line in enumerate(lines):
            if "erg/sec/cm" in line:
                star_line = i + 1
                break

        df = pd.read_csv(
            "run/{0}/lcout.active".format(self.directory),
            delim_whitespace=True,
            skipinitialspace=True,
            index_col=False,
            skiprows=star_line,
            nrows=2,
            names=["star", "m", "radius", "M_bol", "logg", "L", "Lsun"],
        )
        L1 = self.tf_number(df.iloc[0].L)
        L2 = self.tf_number(df.iloc[1].L)
        self.rad1 = self.tf_number(df.iloc[0].radius)
        self.rad2 = self.tf_number(df.iloc[1].radius)
        self.l_ratio = np.mean(L1 / (L1 + L2))

        fit_list = list()
        for i in range(len(mpage_list)):
            if mpage_list[i] == 1:
                for j, line in enumerate(lines[start_line_list[i]: start_line_list[i] + 100]):
                    if line == "      JD               Phase     light 1       light 2       (1+2+3)      norm lite      sep/a   magnitude  magnitude      (days)\n":
                        line_df = j + 1
                        break
                df = pd.read_csv(
                    "run/{0}/lcout.active".format(self.directory),
                    delim_whitespace=True,
                    skipinitialspace=True,
                    index_col=False,
                    skiprows=start_line_list[i] + line_df,
                    nrows=100,
                    names=[
                        "jd",
                        "phase",
                        "l1",
                        "l2",
                        "lc",
                        "ln",
                        "sep",
                        "mag",
                        "magd",
                        "time",
                    ],
                )
                fit_list.append(df)
            elif mpage_list[i] == 2:
                for j, line in enumerate(lines[start_line_list[i]: start_line_list[i] + 100]):
                    if line == "      JD              Phase     V Rad 1     V Rad 2     del V1      del V2      V1 km/s        V2 km/s         (days)\n": 
                        line_df = j + 1
                        break
                df = pd.read_csv(
                    "run/{0}/lcout.active".format(self.directory),
                    delim_whitespace=True,
                    skipinitialspace=True,
                    index_col=False,
                    skiprows=start_line_list[i] + line_df,
                    nrows=100,
                    names=[
                        "jd",
                        "phase",
                        "vrad1",
                        "vrad2",
                        "dvrad1",
                        "dvrad2",
                        "v1",
                        "v2",
                        "time",
                    ],
                )
                df.v1 *= self.VUNIT
                df.v2 *= self.VUNIT
                self.fit_rv = df

        self.fit = fit_list

    @staticmethod
    def tf_number(input_string):
        """Transform the input string from lcout into float
        
        Args:
            input_string (string): input string
        """
        try:
            float(input_string)
        except:
            if "*" not in input_string:
                value = float(re.sub("[Dd]", "e", input_string))
            else:
                value = np.nan
        else:
            value = float(input_string)
        return value

    def clean_run(self):
        subprocess.call("rm -rf run/{0}".format(self.directory), shell=True)

    def print_info(self):
        print("Name: {0}".format(self.directory))
        print("Period: {0:.6f}".format(self.PERIOD))
        print("Q: {0:.2f}".format(self.q_best))
        print("i: {0:.2f}".format(self.XINCL))
        print("T: {0:.0f} {1:.0f}".format(self.TAVH * 1e4, self.TAVC * 1e4))
        print("Omega: {0:.2f} {1:.2f}".format(self.PHSV, self.PCSV))
        if hasattr(self, "lp"):
            print("L: {0:.2f} {1:.2f}".format(self.lp, self.ls))
        if hasattr(self, "mp"):
            print("mass: {0:.2f} {1:.2f}".format(self.mp, self.ms))
        if hasattr(self, "radp"):
            print("rad: {0:.2f} {1:.2f}".format(self.radp, self.rads))

    def clean_active(self):
        filename_list = ["dcin.active", "dcout.active", "lcin.active", "lcout.active", "lcin.input_from_dc"]
        for filename in filename_list:
            p = Path("run/{0}/{1}".format(self.directory, filename))
            if p.is_file():
                p.unlink()

    def cal_fit(self):
        if self.q_best < 1:
            self.pri_num = 1
            self.sec_num = 2
        else:
            self.pri_num = 2
            self.sec_num = 1
        self.is_final = False
        A_default = 2
        if self.run_dc(self.q_best, A=A_default, alarm_time=200):
            self.run_lc()
            self.read_lcout()
            self.cal_error(self.q_best)

        self.lp = getattr(self, "l{0}_bol".format(self.pri_num))
        self.ls = getattr(self, "l{0}_bol".format(self.sec_num))
        self.mp = 10 ** ((np.log10(self.lp / 10 ** -0.2)) / 4.216)
        self.ms = self.mp * (self.q_best if self.pri_num == 1 else 1 / self.q_best)
        self.A = ((self.mp + self.ms) * self.PERIOD ** 2 * 74.5374153813103) ** (
            1 / 3
        )  # 74.5 is G/(2pi)^2
        self.radp = getattr(self, "rad{0}".format(self.pri_num)) / A_default * self.A
        self.rads = getattr(self, "rad{0}".format(self.sec_num)) / A_default * self.A

    def correct_temp_comb(self):
        temp_comb = (
            (self.l1_bol + self.l2_bol)
            / (self.l1_bol / self.TAVH ** 4 + self.l2_bol / self.TAVC ** 4)
        ) ** 0.25
        alpha = temp_comb / self.temp_color
        self.TAVH /= alpha
        self.TAVC /= alpha

    def clean_lc(self, threshold=1.7):
        for i, lc in enumerate(self.lc):
            x = lc.data.phase
            y = lc.data.mag
            if "mag_err" in lc.data.columns:
                y_err = lc.data.mag_err
            else:
                y_err = np.ones_like(x) * 0.1
            mtf = MultiTermFit(2 * np.pi, 8)
            mtf.fit(x, y, y_err)
            X_scaled = mtf._make_X(x)
            y_fit = np.dot(X_scaled, mtf.w_)
            v = y - y_fit
            v_mean = np.mean(v)
            v_std = np.std(v)
            flag = np.abs(v - v_mean) / v_std < threshold
            self.lc[i].data = self.lc[i].data.assign(flag=flag)

    def clean_lc_gp(self, threshold=1.7):
        """Clean the light curve using Gaussian Process (GP)

        Using Constant + RationalQuadratic + WhiteKernel

        Args:
            threshold (float): threshold for rejection
                               (1.7)

        """
        for i, lc in enumerate(self.lc):
            x = lc.data.phase.values
            y = lc.data.mag.values
            X = x.reshape(-1, 1)
            kernel = ConstantKernel() + Matern(length_scale=2, nu=3 / 2) + WhiteKernel()
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            gp.fit(X, y)
            y_pred, std = gp.predict(X, return_std=True)
            flag = np.abs(y - y_pred) / std < threshold
            self.lc[i].data = self.lc[i].data.assign(flag=flag)

    def cal_bol_mag_abs(self, dm):
        self.dm = dm
        if self.run_dc(self.q_best, alarm_time=200):
            self.run_lc()
            self.read_lcout()
            if len(self.lc) == 1:
                mag_abs = np.min(self.fit[0].magd) - self.dm - self.lc[0].AEXTINC
            else:
                for i, lc in enumerate(self.lc):
                    if lc.IBAND == 7:
                        mag_abs = np.min(self.fit[i].magd) - self.dm - lc.AEXTINC
            self.mag_abs = mag_abs
            l1 = 10 ** (self.mag_abs / -2.5) * self.l_ratio
            l2 = 10 ** (self.mag_abs / -2.5) * (1 - self.l_ratio)
            self.mag1_abs = np.log10(l1) * -2.5
            self.mag2_abs = np.log10(l2) * -2.5
            self.BC1 = self.cal_bc(self.TAVH * 1e4)
            self.BC2 = self.cal_bc(self.TAVC * 1e4)
            self.mag1_abs_bol = self.mag1_abs + self.BC1
            self.mag2_abs_bol = self.mag2_abs + self.BC2
            self.l1_bol = 10 ** ((self.mag1_abs_bol - 4.75) / -2.5)
            self.l2_bol = 10 ** ((self.mag2_abs_bol - 4.75) / -2.5)
            self.mag_abs_bol = -2.5 * np.log10(self.l1_bol + self.l2_bol) + 4.75

    def cal_bc(self, T):
        table = Table.read("material/Pecaut.fit").to_pandas()
        bc = np.interp(T, table["Teff"][::-1], table["BCV"][::-1])
        return bc


if __name__ == "__main__":
    init_run()
