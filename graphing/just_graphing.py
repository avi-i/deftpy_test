from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from adjustText import adjust_text

df_cf = pd.read_csv('witman_data_ternary.csv')
dropped_values = df_cf[df_cf.isnull().any(axis=1)]
dropped_values.to_csv("weighted_drops")
exit(4)
df_cf = df_cf.dropna()
cfm = HuberRegressor()
X = df_cf[["Vr_max"]]
#X = df_cf[["Eb_sum", "Vr_max"]]
#X = df_cf[["Vr_max", "Eg"]]
X = df_cf[["Eb_sum", "Vr_max", "Eg"]]
#X = df_cf[["Eb_sum", "Vr_max", "Eg", "Ehull"]]
y = df_cf["Ev"]
cfm.fit(X, y)
y_pred = cfm.predict(X)
coefs = cfm.coef_

plt.style.use('seaborn-talk')
plt.scatter(y_pred, y)
plt.plot([1, 9], [1, 9], "k--")
plt.xlim(min(y_pred) - 1, max(y_pred) + 1)
plt.ylim(min(y) - 1, max(y) + 1)
plt.gca().set_aspect('equal', adjustable='box')
#equation = f"$E_v$ =  {cfm.coef_[0]:.2f} $V_r$ + {cfm.intercept_:.2f} (eV)"
#equation = f"$E_v$ = {cfm.coef_[0]:.2f} $\\Sigma E_b$ {cfm.coef_[1]:.2f} $V_r$ + {cfm.intercept_:.2f} (eV)"
equation = f"$E_v$ = {cfm.coef_[0]:.2f} $\\Sigma E_b$ {cfm.coef_[1]:.2f} $V_r$ + {cfm.coef_[2]:.2f} $E_g$ + {cfm.intercept_:.2f} (eV)"
#equation = f"$E_v$ = {cfm.coef_[0]:.2f} $\\Sigma E_b$ {cfm.coef_[1]:.2f} $V_r$ + {cfm.coef_[2]:.2f} $E_g$ + {cfm.coef_[3]:.2f} $E_{{hull}}$ + {cfm.intercept_:.2f} (eV)"
mae = np.mean(np.abs(y - y_pred))
plt.text(1.7, 6, f"MAE = {mae:.2f} eV", fontsize=14)
#binary
oxides = f"$MO_x$"
#binary and ternary
#oxides = f"$MO_x$, $ABO_x$"
plt.text(1.7, 6.5, oxides, fontsize=14)
#add number of data points as text
plt.text(1.7, 5.5, f"n = {len(y)}", fontsize=14)
texts = []
for x, y, s in zip(y_pred, y, df_cf["formula"]):
        texts.append(plt.text(x, y, s, size=10))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
plt.xlabel(str(equation))
plt.ylabel(f"Witman et al. $E_v$")
plt.title("CFM for binary oxides with band gap energies")
#plt.legend(bbox_to_anchor=(1.1, 1.0), prop={'size': 8})
#plt.text(1.1, 0.9, "each color represents a unique defect id")
plt.savefig("CFM_b_Eb_Vr_Eg_nw.png", dpi=300)
