from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import autograd.numpy as np
from autograd.numpy import multiply as ewm

from model_full import *
# from initial_model import Model

n = number_group

data = np.load(savefilename+".npz", allow_pickle=True)

configurations = data["configurations"]
y0, t_total, N_total, number_group, population_proportion, \
t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

simulation_first_confirmed=data["simulation_first_confirmed"]
solution=data["solution"]
controls=data["controls"]
solution_opt=data["solution_opt"]
controls_opt=data["controls_opt"]

alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = controls_opt

# print("controls", controls_opt[3:])

model = Misfit(configurations, parameters, controls)

model.plotsolution(misfit.t_total, solution, solution_opt, filename_prex)
plt.close('all')

solution_opt = model.grouping(solution_opt)
simulation_first_hospitalized = simulation_first_confirmed + model.lag_hospitalized
simulation_first_deceased = simulation_first_confirmed + model.lag_deceased

plt.figure()

# date_confirmed = range(simulation_first_confirmed, simulation_first_confirmed+len(data_confirmed))
# plt.plot(date_confirmed, data_confirmed, "y.-", label="data confirmed")
#
# plt.plot(solution_opt[:, 8], "y-", label="simulation confirmed")

date_hospitalized = range(simulation_first_hospitalized, simulation_first_hospitalized+len(model.data_hospitalized))
plt.plot(date_hospitalized, model.data_hospitalized, "r.-", label="data hospitalized")
plt.plot(solution_opt[:, 5], "r-", label="simulation hospitalized")

date_deceased = range(simulation_first_deceased, simulation_first_deceased+len(model.data_deceased))
plt.plot(date_deceased, model.data_deceased, "k.-", label="data deceased")
plt.plot(solution_opt[:, 7], "k-", label="simulation deceased")

plt.xlabel("time (days)")
plt.ylabel("number of cases")
plt.legend()
plt.grid()
filename = filename_prex + "data_comparison.pdf"
plt.savefig(filename)

plt.figure()

# date_confirmed = range(simulation_first_confirmed, simulation_first_confirmed+len(data_confirmed))
# plt.plot(date_confirmed[1:], np.diff(data_confirmed), "y.-", label="data confirmed")
#
# plt.plot(np.diff(solution_opt[:, 8]), "y-", label="simulation confirmed")

date_hospitalized = range(simulation_first_hospitalized, simulation_first_hospitalized+len(model.data_hospitalized))
plt.plot(date_hospitalized, model.data_hospitalized, "r.-", label="data hospitalized")
plt.plot(solution_opt[:, 5], "r-", label="simulation hospitalized")

date_deceased = range(simulation_first_deceased, simulation_first_deceased+len(model.data_deceased))
plt.plot(date_deceased[1:], np.diff(model.data_deceased), "k.-", label="data deceased")
plt.plot(np.diff(solution_opt[:, 7]), "k-", label="simulation deceased")

plt.xlabel("time (days)")
plt.ylabel("number of cases")
plt.legend()
plt.grid()
filename = filename_prex + "data_comparison_daily.pdf"
plt.savefig(filename)
# plt.show()

# Rt = data["Rt"]
# Rt_opt = data["Rt_opt"]
#
# plt.figure()
# plt.plot(Rt, "b.-", label="before inference")
# plt.plot(Rt_opt, "r.-", label="after inference")
# plt.legend()
# plt.xlabel("time (days)")
# plt.ylabel("effective reproduction number")
# plt.grid()
# filename = filename_prex + "reproduction.pdf"
# plt.savefig(filename)

# plt.figure()
# plt.plot(controls_opt, '.-')
# plt.show()

# time_delta = timedelta(days=1)
#
# start_date = datetime(2020, 2, 15)
# stop_date = start_date + timedelta(len(t))
# dates = mdates.drange(start_date, stop_date, time_delta)
#
# start_date = start_date + timedelta(np.int(ten_death_day))
# stop_date = start_date + timedelta(len(t))
# dates_control = mdates.drange(start_date, stop_date, time_delta)

labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

if model_type is "vector":

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height > 1:
                height = np.int(height)

            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)


    # ############### plot age and risk distribution
    # age_distribution = np.array([0.14, 0.15, 0.14, 0.14, 0.13, 0.12, 0.1, 0.05, 0.03])
    # high_risk_percentage_of_population = np.array(
    #     [0.11321124, 0.12544888, 0.14938879, 0.18503098, 0.23237544, 0.29142218, 0.36217119, 0.44462247, 0.53877604])
    # high_risk = np.around(ewm(age_distribution, high_risk_percentage_of_population), decimals=2)
    # low_risk = np.around(ewm(age_distribution, 1-high_risk_percentage_of_population), decimals=2)
    #
    # plt.close('all')
    #
    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width / 2, low_risk, width, label='low risk')
    # rects2 = ax.bar(x + width / 2, high_risk, width, label='high risk')
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('percentage of different groups', fontsize=16)
    # ax.set_title("Prior information: age and risk distribution", fontsize=16)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, fontsize=10)
    # ax.legend(fontsize=16)
    #
    # autolabel(rects1)
    # autolabel(rects2)
    #
    # fig.tight_layout()
    #
    # plt.savefig("figure/age-risk-18.pdf")
    #
#

    ############# plot deceased cases ################
    titles = ["# deaths w/o control", "# deaths w control"]
    filenames = [filename_prex+"death_without_control.pdf", filename_prex+"death_with_control.pdf"]
    solution = data["solution"]
    solution_opt=data["solution_opt"]
    today = data["today"]
    deceased = [solution[today, 7*number_group:8*number_group]]
    deceased.append(solution_opt[today, 7*number_group:8*number_group])
    for i in range(2):
        plt.close('all')
        cont_i = deceased[i]
        low_risk = np.ceil(cont_i[:9])
        high_risk = np.ceil(cont_i[9:])

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, low_risk, width, label='low risk')
        rects2 = ax.bar(x + width/2, high_risk, width, label='high risk')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('# deaths', fontsize=16)
        title = titles[i]+", total = "+str(np.int(np.sum(deceased[i])))
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=16)

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.savefig(filenames[i])
        # plt.show()
        # plt.close()



########### plot the control measures
titles = ["social distancing by age and risk", "quarantine by age and risk",
          "isolation ratio by age and risk"]
filenames = [filename_prex+"social-distancing.pdf", filename_prex+"quarantine.pdf", filename_prex+"isolation.pdf",
             filename_prex + "HFR.pdf", filename_prex + "kappa.pdf"]

# for i in range(4):
#     plt.close('all')
#     cont_i = controls_opt[i]
#     for j in range(len(cont_i)):
#         low_risk = np.around(cont_i[j][:5], decimals=2)
#         high_risk = np.around(cont_i[j][5:], decimals=2)
#
#         x = np.arange(len(labels))  # the label locations
#         width = 0.35  # the width of the bars
#
#         fig, ax = plt.subplots()
#         rects1 = ax.bar(x - width/2, low_risk, width, label='low risk')
#         rects2 = ax.bar(x + width/2, high_risk, width, label='high risk')
#
#         # Add some text for labels, title and custom x-axis tick labels, etc.
#         ax.set_ylabel('control strength', fontsize=16)
#         ax.set_title(titles[i], fontsize=16)
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, fontsize=16)
#         ax.set_ylim(0., 1.)
#         ax.legend(fontsize=16)
#
#         autolabel(rects1)
#         autolabel(rects2)
#
#         fig.tight_layout()
#         filename = filenames[i] + "_" + str(j) + ".pdf"
#         plt.savefig(filename)
#         plt.close()

control_variable = ["social distancing reduction", "quarantine ratio by contact tracing", "isolation ratio by testing",
                    "hospitalized fatality ratio", "infection (confirmed) hospitalization ratio"]
# for i in range(4):
#     plt.close('all')
#     plt.figure()
#     cont_i = controls_opt[i]
#     day = np.array(range(len(cont_i))) * number_days_per_control_change + ten_death_day
#
#     day = np.repeat(day, 2)
#     day[1::2] += number_days_per_control_change
#     cont_i = np.repeat(cont_i, 2, axis=0)
#
#     plt.plot(day, cont_i[:, 0], 'c.--', label='low risk, age in [0, 4]')
#     plt.plot(day, cont_i[:, 1], 'go--', label='low risk, age in [5, 17]')
#     plt.plot(day, cont_i[:, 2], 'kx--', label='low risk, age in [18, 49]')
#     plt.plot(day, cont_i[:, 3], 'rd--', label='low risk, age in [50, 64]')
#     plt.plot(day, cont_i[:, 4], 'bs--', label='low risk, age in [65+]')
#     plt.plot(day, cont_i[:, 5], 'c.-', label='high risk, age in [0, 4]')
#     plt.plot(day, cont_i[:, 6], 'go-', label='high risk, age in [5, 17]')
#     plt.plot(day, cont_i[:, 7], 'kx-', label='high risk, age in [18, 49]')
#     plt.plot(day, cont_i[:, 8], 'rd-', label='high risk, age in [50, 64]')
#     plt.plot(day, cont_i[:, 9], 'bs-', label='high risk, age in [65+]')
#
#     plt.title(control_variable[i])
#     plt.legend(loc='best')
#     plt.xlabel('time t (days)')
#     plt.ylabel('control strength')
#     plt.grid()
#     filename = filenames[i] + "_change.pdf"
#     plt.savefig(filename)


# for i in range(4):
#     plt.close('all')
#     plt.figure()
#     cont_i = controls_opt[i]
#     day = np.array(range(len(cont_i))) * number_days_per_control_change
#
#     day = np.repeat(day, 2)
#     day[1::2] += number_days_per_control_change
#     cont_i = np.repeat(cont_i, 2, axis=0)
#
#     for j in range(9):
#         ax = plt.subplot(3, 3, j+1)
#         label_low = "low " + labels[j]
#         label_high = "high " + labels[j]
#         plt.plot(day, cont_i[:, j], '.--', label=label_low)
#         plt.plot(day, cont_i[:, j+9], '.:', label=label_high)
#
#         if j != 7:
#             plt.setp(ax.get_xticklabels(), visible=False)
#         else:
#             plt.setp(ax.get_xticklabels(), fontsize=6)
#         plt.setp(ax.get_yticklabels(), fontsize=6)
#
#         plt.legend(loc="best", fontsize=6)
#         plt.grid(True)
#         plt.ylim(0, 1)
#         if j == 1:
#             plt.title(control_variable[i], fontsize=10)
#         if j == 7:
#             plt.xlabel('time t (days)', fontsize=10)
#
#     filename = filenames[i] + "_change.pdf"
#
#     plt.savefig(filename)


for i in range(number_time_dependent_controls):
    plt.close('all')
    plt.figure()
    cont_i = model.interpolation(model.t_total, model.t_control, controls_opt[i])

    # plt.figure()
    # plt.plot(cont_i, '.')
    #
    # # plt.legend(loc="best", fontsize=16)
    # plt.grid(True)
    # if i != 3:
    #     plt.ylim(0, 1)
    # plt.xlabel('time t (days)', fontsize=16)

    if model_type is "scalar" or model_type is "vector":
        plt.figure()
        plt.plot(cont_i, '.')

        # plt.legend(loc="best", fontsize=16)
        plt.grid(True)
        if i < 3:
            plt.ylim(0, 1)
        plt.xlabel('time t (days)', fontsize=16)
    else:
        for j in range(9):
            ax = plt.subplot(3, 3, j+1)
            label_low = "low " + labels[j]
            label_high = "high " + labels[j]
            plt.plot(cont_i[:, j], '.', label=label_low)
            plt.plot(cont_i[:, j+9], 'r.', label=label_high)

            if j != 7:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.setp(ax.get_xticklabels(), fontsize=6)
            plt.setp(ax.get_yticklabels(), fontsize=6)

            plt.legend(loc="best", fontsize=6)
            plt.grid(True)
            if i != 3:
                plt.ylim(0, 1)
            if j == 1:
                plt.title(control_variable[i], fontsize=10)
            if j == 7:
                plt.xlabel('time t (days)', fontsize=10)

    filename = filenames[i]

    plt.savefig(filename)

plt.close("all")

parameter_tick = [r'$\alpha', r'$q$', r'$\tau$', 'HFR', r'$\kappa$', r'$\beta$', r'$\delta$', r'$\sigma$', r'$\eta_I$',
                  r'$\eta_Q$', r'$\mu$', r'$\gamma_I$', r'$\gamma_A$', r'$\gamma_H$', r'$\gamma_Q$']

if model_type is "scalar":
    plt.figure()
    plt.plot(controls[number_time_dependent_controls:], 'x', label='mean')
    plt.plot(controls_opt[number_time_dependent_controls:], 'o', label='optimal')
    plt.legend()
    tick = parameter_tick[number_time_dependent_controls:]
    plt.xticks(range(len(tick)), tick)
    plt.grid()
    plt.ylim(0, 2)
else:
    for j in range(9):

        ax = plt.subplot(3, 3, j + 1)

        label_low = "opt low " + labels[j]
        label_high = "opt high " + labels[j]
        length_controls = len(controls[number_time_dependent_controls:])
        xl = [controls_opt[i][j] for i in range(number_time_dependent_controls, number_time_dependent_controls+length_controls)]
        xh = [controls_opt[i][j+9] for i in range(number_time_dependent_controls, number_time_dependent_controls+length_controls)]
        plt.plot(xl, 'x', label=label_low)
        plt.plot(xh, 'o', label=label_high)

        label_low = "mean low " + labels[j]
        label_high = "mean high " + labels[j]
        xl = [parameters[i][j] for i in range(len(parameters))]
        xh = [parameters[i][j+9] for i in range(len(parameters))]
        plt.plot(xl, 'x', label=label_low)
        plt.plot(xh, 'o', label=label_high)

        # if j != 7:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        # else:
        plt.setp(ax.get_xticklabels(), fontsize=6)
        plt.setp(ax.get_yticklabels(), fontsize=6)

        plt.legend(loc="best", fontsize=6)
        plt.grid(True)
        plt.ylim(0, 1)
        if j == 1:
            plt.title("estimated parameter values", fontsize=10)
        # if j == 7:
        tick = parameter_tick[number_time_dependent_controls:]
        plt.xticks(range(len(tick)), tick)

plt.savefig(filename_prex+"other-parameters.pdf")
plt.close("all")

# total confirmed and unconfirmed
solution_opt=data["solution_opt"]
solution_opt = model.grouping(solution_opt)

plt.figure()
plt.semilogy(solution_opt[:, -2], 'r.-', label="confirmed")
plt.semilogy(solution_opt[:, -1], 'y.-', label="unconfirmed")
ratio = np.divide(solution_opt[:, -1], solution_opt[:, -2])
plt.semilogy(ratio, 'k.-', label="ratio")
plt.legend()
plt.grid()
plt.xlabel('time t (days)')
plt.ylabel("the number of infections")
plt.title("unconfirmed/confirmed = " + str(np.around(ratio[-1], decimals=1)))
plt.savefig(filename_prex+"confirmed-unconfirmed.pdf")


lag_death = np.int(np.mean(np.divide(1., eta_I) + np.divide(1., mu)))
IFR = np.divide(solution_opt[date_deceased[lag_death:], 7],
      (solution_opt[simulation_first_deceased:simulation_first_deceased+len(date_deceased[lag_death:]), 8]
        + solution_opt[simulation_first_deceased:simulation_first_deceased+len(date_deceased[lag_death:]), 9]))

plt.figure()
plt.plot(date_deceased[lag_death:], IFR, 'r.-')
plt.grid()
plt.xlabel('time t (days)')
plt.ylabel("IFR")
plt.savefig(filename_prex+"IFR.pdf")

print("State = ", state_name)